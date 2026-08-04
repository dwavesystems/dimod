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
        'test_quadratic_model.cpp',
        'test_utils.cpp',
    ],
    dependencies: [
        catch2,
        dimod_dep,
    ],
)

test('tests-all', tests_all, verbose: true)
