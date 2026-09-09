project(
    'dimod',
    'c', 'cpp', 'cython',
    default_options: [
        'cpp_std=c++11',
        'buildtype=release',
        'b_ndebug=if-release',  # add -DNDEBUG when building for "release", which we do by default
        'optimization=3',
        'pkgconfig.relocatable=true',
        'warning_level=1',  # equivalent to -Wall. We should consider cranking it up to 3
        'python.allow_limited_api=false',  # Our build system can then opt-in for Python 3.12+
    ],
    meson_version: '>= 1.10.0',
    # As of July 2026, there is no nicer way that I know of to dynamically get the version
    version: run_command(
        ['dimod/_build/_version.py'],
        capture: true,
        check: true,
    ).stdout(),
)

cpp = meson.get_compiler('cpp')
py = import('python').find_installation(pure: false)

if cpp.has_argument('-g1')
  add_project_arguments('-g1', language: 'cpp')
endif

# We build the limited API when we have Python>=3.12, which is the lowest
# version that Cython makes use of vectorcall.
# See https://cython.readthedocs.io/en/stable/src/userguide/limited_api.html
# By default this will do nothing unless the user overrides the
# python.allow_limited_api=false we specify above, which we do when building
# with cibuildwheel
if py.version().version_compare('>=3.12')
    limited_api = '3.12'
else
    limited_api = ''  # i.e. no limited api
endif

dimod_dep = declare_dependency(
    include_directories: 'dimod/include/',
)

# Some packages that use dimod as a subproject do not need the extension modules
# and it speeds up the build a lot to be able to turn them off.
if get_option('build-python')
    foreach path : [
        'dimod/binary/cybqm/cybqm_float32.pyx',
        'dimod/binary/cybqm/cybqm_float64.pyx',
        'dimod/constrained/cyconstrained.pyx',
        'dimod/constrained/cyexpression.pyx',
        'dimod/cyqmbase/cyqmbase_float32.pyx',
        'dimod/cyqmbase/cyqmbase_float64.pyx',
        'dimod/discrete/cydiscrete_quadratic_model.pyx',
        'dimod/quadratic/cyqm/cyqm_float32.pyx',
        'dimod/quadratic/cyqm/cyqm_float64.pyx',
        'dimod/cyutilities.pyx',
        'dimod/cyvariables.pyx',
    ]
        basename = path.split('/')[-1].split('.')[0]
        directory = '/'.join(path.split('/').slice(0, -1))

        py.extension_module(
            basename,
            path,
            cython_args : ['-Xfreethreading_compatible=True'],
            dependencies: dimod_dep,
            install: true,
            override_options : ['cython_language=cpp'],
            subdir: directory,
            limited_api: limited_api,
        )
    endforeach

    py.extension_module(
        'cylp',
        'dimod/cylp.pyx',
        cython_args : ['-Xfreethreading_compatible=True'],
        dependencies: [
            dimod_dep,
            declare_dependency(
                include_directories: 'dimod/_extern',
                sources: 'dimod/_extern/filereaderlp/reader.cpp',
            ),
        ],
        install: true,
        override_options : ['cython_language=cpp'],
        subdir: 'dimod/',
        limited_api: limited_api,
    )

    # If we're not building the extensions, we shouldn't install the package
    # because imports will fail.
    install_subdir('dimod', install_dir: py.get_install_dir(pure: false))

endif  # get_option('build-python')

# meson doesn't disable tests by default, so we let the user decide via the build-tests
# option. If set to 'auto', then we build the tests for non-release builds.
if get_option('build-tests').enabled() or (get_option('build-tests').auto() and get_option('buildtype') != 'release')
    subdir('testscpp/')
endif
