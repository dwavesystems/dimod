# Notes on style

* Use [Declaration Order](https://google.github.io/styleguide/cppguide.html#Declaration_Order).
* For getters/setters:
    * `attribute()` should return a value or a const reference
    * `set_attribute()` should set the attribute
    * If a non-const reference is desired, should use `[]` or `attribute_ref()`.
