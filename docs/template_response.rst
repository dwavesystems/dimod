Responses
=========

.. currentmodule:: dimod

.. automodule:: dimod.responses

SpinResponse
------------

.. autoclass:: SpinResponse

.. autosummary::
   :toctree: generated/

   SpinResponse.samples
   SpinResponse.energies
   SpinResponse.items
   SpinResponse.add_sample
   SpinResponse.add_samples_from
   SpinResponse.relabel_samples
   SpinResponse.as_binary

BinaryResponse
--------------

.. autoclass:: BinaryResponse

.. autosummary::
   :toctree: generated/

   BinaryResponse.samples
   BinaryResponse.energies
   BinaryResponse.items
   BinaryResponse.add_sample
   BinaryResponse.add_samples_from
   BinaryResponse.relabel_samples
   BinaryResponse.as_spin

TemplateResponse
----------------

You can also create your own response object inheriting from the TemplateResponse class.

.. autoclass:: TemplateResponse

.. autosummary::
   :toctree: generated/

   TemplateResponse.samples
   TemplateResponse.energies
   TemplateResponse.items
   TemplateResponse.add_sample
   TemplateResponse.add_samples_from
   TemplateResponse.relabel_samples
