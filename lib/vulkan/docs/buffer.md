Think of a buffer as a file stream. treat it as an inefficient memory recource.

to create a buffer:
```cpp
#include <gsplat/vulkan/memory.h>
using namespace gsplat::vulkan;

buffer my_buffer{12}; // buffer of 12 bytes.
buffer my_buffer{3 * sizeof(int)}; // buffer of 3 ints.
```

to auto convert from/to vector<T>:
```cpp
#include <gsplat/vulkan/memory.h>
using namespace gsplat::vulkan;

std::vector<float> vec{};
buffer my_buffer = co_await buffer::from_vector(vec);
std::vector<float> vec2 = co_await my_buffer.to_vector<float>(); // must specify value type.
```

buffers can be moved:
```cpp
#include <gsplat/vulkan/memory.h>
using namespace gsplat::vulkan;

buffer buf1;
buffer buf2 = std::move(buf1);
// cant do:
// buffer buf3 = buf1;
```

buffers can be copied:
```cpp
#include <gsplat/vulkan/memory.h>
using namespace gsplat::vulkan;

buffer buf1;
buffer buf2 = co_await buf1.copy();
```

to use coroutines, just use `co_await` for awaiters (coroutines, functions that return coroutines, functions that return awaiters)
you must use `co_return` instead of return.

it is more efficient to `co_guard` to ensure stack allocation:
```cpp
coroutine<void> my_co();

co_await co_guard{my_co()};
```
although the compiler may detect stack allocation anyways. so this is an optional optimization.

implicit start on co_await:
coroutines dont initially run when first invoked. you must call resume at least once.
but, if you dont resume a coroutine (the coroutine has not started), then the coroutine will resume on `co_await`.
if you do not want this behavior (such as enqueing on a different thread), then call `my_coro.prevent_implicit_start();`