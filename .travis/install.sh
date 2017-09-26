#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on OS X
    brew install pyenv-virtualenv

    case "${TOXENV}" in
        py27)
            # Install some custom Python 2.7 requirements on OS X
                pip install requirements.txt
            ;;
        py36)
            # Install some custom Python 3.6 requirements on OS X
                pip install requirements.txt
            ;;
        pynightly)
            # Install some custom Python nightly requirements on OS X
                pip install requirements.txt
            ;;

    esac
else
    pip install requirements.txt
fi
