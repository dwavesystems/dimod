catch2 = subproject(
    'catch2',
).get_variable('catch2_dep')

tests_all = executable(
    'tests-all',
    [
        'test_main.cpp',
        'test_binary_quadratic_model.cpp',
        'test_constrained_quadratic_model.cpp',
        'test_iterators.cpp',
        'test_presolve.cpp',
        'test_presolveimpl.cpp',
        'test_quadratic_model.cpp',
        'test_roof_duality.cpp',
        'test_utils.cpp',
    ],
    dependencies: [
        catch2,
        dimod_dep,
    ],
    include_directories: '../dimod/src',
)

test('tests-all', tests_all, verbose: true)
