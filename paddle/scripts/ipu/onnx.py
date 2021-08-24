# Copyright (c) 2020 Graphcore Ltd. All rights reserved
import logging
import os
import re
import csv
from ctypes.util import find_library
import clang.cindex

current_dir = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger('OnnxParser')

popart_include_dir = None
name_data = []

popart_files = ["builder.hpp", "builder.h.gen"]

nodeBlacklist = {"DomainOpSet", "Builder", "getOpsetVersion", "AiOnnxOpset11"}


def find_popart_includes():
    assert "CONDA_PREFIX" in os.environ, ("You need to run this script from "
                                          "inside an activated buildenv")
    expected_path = os.path.realpath(
        os.path.join(os.environ["CONDA_PREFIX"], "..", "popart"))
    assert os.path.isdir(expected_path), ("You need to configure your build "
                                          "by running cmake")
    return expected_path


def init(popart_path=None, clang_path=None):
    builder_path = os.path.isfile(
        os.path.join(popart_path, "popart", "builder.hpp"))
    # print(popart_path)
    assert builder_path, ("Unable to locate popART's popart/builder.hpp "
                          "in " + popart_path)
    global popart_include_dir
    popart_include_dir = popart_path

    print("Will pick up popART headers from: {0}".format(popart_include_dir))
    for (i, fname) in enumerate(popart_files):
        popart_files[i] = os.path.join(popart_include_dir, "popart", fname)

    if clang.cindex.Config.loaded:
        # Already initialised
        return

    if clang_path is None:
        for version in [11, 9, 8, 7, 6]:
            logger.debug('Trying to find: clang-%s', str(version))
            clang_path = find_library('clang-' + str(version))
            if clang_path is not None:
                break

    assert clang_path is not None, 'Could not find clang'
    logger.info('Will use clang: %s', clang_path)
    clang.cindex.Config.set_library_file(clang_path)


def find_functions(jsonOutput, node, namespace=""):
    # If this is not the file path provided on the comand line, skip.
    if node.location.file is not None and str(
            node.location.file) not in popart_files:
        return
    if node.spelling in nodeBlacklist:
        return

    if node.kind == clang.cindex.CursorKind.CLASS_DECL:
        namespace = node.spelling

    if node.kind != clang.cindex.CursorKind.CXX_METHOD:
        for child in node.get_children():
            find_functions(jsonOutput, child, namespace)
        return

    functionName = node.spelling
    returnType = str(node.type.spelling).split("(")[0]
    operation = dict()
    operation["type"] = returnType
    operation["args"] = []

    if node.access_specifier != clang.cindex.AccessSpecifier.PUBLIC:
        return

    argNum = 0
    for child in node.get_children():
        argument = {}
        if child.kind != clang.cindex.CursorKind.PARM_DECL:
            continue

        argument["type"] = child.type.spelling
        argument["name"] = child.spelling

        # skip 'name' argument
        if argument['name'] == 'name':
            continue

        # skip DebugContext argument
        if re.search('DebugContext', argument['type']):
            continue

        argument["num"] = argNum
        operation["args"].append(argument)
        argNum += 1

    if namespace not in jsonOutput:
        jsonOutput[namespace] = {}

    jsonOutput[namespace][functionName] = operation


# parse()
#
# Parse popART header files and extract onnx operator information
# Returns:
#   Map of operators, return types and arguments
def parse():
    index = clang.cindex.Index.create()
    print(" popart_include_dir ", popart_include_dir)
    path = os.path.realpath(
        os.path.join(popart_include_dir, "popart", "builder.hpp"))
    logger.info('Parsing: %s', path)
    tu = index.parse(
        path,
        args=[
            "-std=c++14", "-I" + popart_include_dir, "-DONNX_NAMESPACE=onnx"
        ])

    for diag in tu.diagnostics:
        logger.warning(diag)

    json = dict()
    find_functions(json, tu.cursor)

    classes = []
    for name in json:
        if name.startswith("Ai"):
            classes.append(name)
        else:
            del json[name]

    classes.reverse()
    added_functions = set()

    for opset in classes:
        to_remove = []

        for name in json[opset]:
            if name in added_functions:
                to_remove.append(name)
            else:
                added_functions.add(name)

        for name in to_remove:
            json[opset].pop(name)

    return json


signatures = dict()


def parse_signatures():
    json = parse()
    classes = []
    for classname in json:
        classes.append(classname)
    classes.reverse()

    type_map = {
        'bool': ['cint'],
        'float': ['cfloat'],
        'int64_t': ['clong', 'dimension'],
        'unsigned int': ['cint'],
        'std::string': ['cstr'],
        'std::vector<float>': ['cfloat_list', 'empty_initializer'],
        'std::vector<int64_t>': ['clong_list', 'empty_initializer'],
        'std::vector<std::string>': ['cstr_list', 'empty_initializer'],
        'nonstd::optional<float>': ['cfloat', 'None'],
        'nonstd::optional<int>': ['cint', 'None'],
        'nonstd::optional<int64_t>': ['clong', 'None'],
        'nonstd::optional<std::string>': ['cstr', 'None'],
        'nonstd::optional<std::vector<int64_t> >':
        ['clong_list', 'dimension_list', 'None'],
        'Attributes::Int': ['clong'],
        'Attributes::Ints': ['clong_list', 'empty_initializer'],
        'popart::ReductionType': ['cint', 'reduction'],
        'popart::ScatterReduction': ['cint', 'scatter_reduction'],
        'popart::Builder': 'ignore',
        'popart::ConstVoidData': 'ignore',
        'popart::MultiConvDilations': 'ignore',
        'popart::MultiConvInputs': 'ignore',
        'popart::MultiConvPads': 'ignore',
        'popart::MultiConvStrides': 'ignore',
        'popart::TensorId': 'ignore'
    }

    for classname in classes:
        for op in json[classname]:
            args = json[classname][op]['args']

            arglist = []
            for arg in args:
                name = arg['name']
                ty = arg['type'].replace('const ', '').replace(' &', '')

                if name == 'args':
                    arglist.append('Args')
                    continue
                if ty not in type_map:
                    assert False, "Unsupported type " + ty + \
                        " in onnx.parse_signatures()"

                if type_map[ty] != 'ignore':
                    arglist.append(type_map[ty])

            signatures[op] = arglist
