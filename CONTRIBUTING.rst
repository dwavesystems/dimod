Contributing
============

Ocean's
`contributing <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
section has guidelines for contributing to Ocean packages.

dimod includes some formatting customization in the
`.clang-format <.clang-format>`_ and `setup.cfg <setup.cfg>`_ files.

Release Notes
-------------

dimod makes use of `reno <https://docs.openstack.org/reno/>`_ to manage its
release notes.

When making a contribution to dimod that will affect users, create a new
release note file by running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``.
Remove any sections not relevant to your changes.
Commit the file along with your changes.

See reno's
`user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_ for
details.