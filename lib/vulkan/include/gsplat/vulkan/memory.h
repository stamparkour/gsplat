#ifndef GSPLAT_VULKAN_MEMORY_H
#define GSPLAT_VULKAN_MEMORY_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <cstddef>
#include <string>
#include <algorithm>
#include <vector>
#include <stamp/task/stream.h>

// uses cpp coroutine/awaiters

namespace gsplat::vulkan {
	struct gptr_t {
		void* ptr = nullptr;
		gptr_t() = default;
		explicit gptr_t(void* ptr) : ptr(ptr) {}
		gptr_t(nullptr_t) : gptr_t() {}
		gptr_t(const gptr_t& other) {
			ptr = other.ptr;
		}
		gptr_t& operator =(nullptr_t) {
			ptr = nullptr;
		}
		gptr_t& operator =(const gptr_t& other) {
			ptr = other.ptr;
		}

		operator bool() const {
			return ptr == nullptr;
		}
		auto operator <=>(const gptr_t& other) const {
			return ptr <=> other.ptr;
		}
		bool operator ==(const gptr_t& other) const {
			return ptr == other.ptr;
		}
	};

	struct vulkan_memory_manager {
		struct node_t {
			std::size_t begin;
			std::size_t end;
		};
		std::list<node_t> device_local_allocs{};
		std::list<node_t> host_visibile_allocs{};
		bool initialized = false;


		vulkan_memory_manager() {

		}
		vulkan_memory_manager() {

		}
		void destroy() {

		}
		void init() {
			if (initialized) return;
			initialized = true;

			// can use 'this' pointer because object does not move.
			[this]() -> stamp::task::coroutine<void> {
				co_await vulkan_context::co_on_destroy();
				destroy();
			}().resume();

		}
		gptr_t alloc(std::size_t size, std::size_t align, memory_type) {
			init();
		}
		void free(gptr_t ptr) {

		}
	};

	struct memory_type {
		enum : int {
			device_local,
			host_visible, // assumes highly volatile
		};
	};

	gptr_t alloc_gpu_mem(std::size_t size, std::size_t align, memory_type);
	void free_gpu_mem(gptr_t);

	class buffer;
	class buffer_io_strand : stamp::task::co_stream_strand {
	private:
		buffer* buffer_v;
		bool eof_v = false;
		bool fail_v = false;
		bool bad_v = false;
		std::size_t pos = 0;

		gptr_t host_mem_v = nullptr;
		char* map_buffer_v = nullptr;
		std::size_t buffer_pos = 0;
		std::size_t buffer_size = 512;
	public:
		buffer_io_strand(buffer*);
		~buffer_io_strand();
	protected:
		virtual bool do_eof() const {
			return eof_v;
		}
		virtual bool do_fail() const {
			return fail_v;
		}
		virtual bool do_bad() const {
			return bad_v;
		}
		virtual void clear() {
			eof_v = false;
			fail_v = false;
			bad_v = false;
		}

		virtual std::size_t do_get_ppos() const {
			return pos;
		}
		virtual void do_set_ppos(std::size_t pos, seekdir dir) {
			switch (dir) {
			case seekdir::beg:
				this->pos = pos;
				break;
			case seekdir::cur:
				this->pos += pos;
				break;
			case seekdir::end:
				this->pos = pos + buffer_v->size();
				break;
			}
			if (this->pos > buffer_v->size()) {
				this->pos = buffer_v->size();
			}
		}
		virtual std::size_t do_get_gpos() const {
			return pos;
		}
		virtual void do_set_gpos(std::size_t pos, seekdir dir) {
			switch (dir) {
			case seekdir::beg:
				this->pos = pos;
				break;
			case seekdir::cur:
				this->pos += pos;
				break;
			case seekdir::end:
				this->pos = pos + buffer_v->size();
				break;
			}
			if (this->pos > buffer_v->size()) {
				this->pos = buffer_v->size();
			}
		}

		virtual std::size_t do_try_put_span(const char* span, std::size_t size) {
			eof_v = false;
			std::ptrdiff_t buf_offset = pos - buffer_pos;
			if (buf_offset < 0 || buf_offset >= buffer_size) return 0;
			std::ptrdiff_t length = std::min(
				(std::ptrdiff_t)size, 
				(std::ptrdiff_t)(buffer_size - buf_offset), 
				(std::ptrdiff_t)(buffer_v->size() - pos));
			stamp::core::s_assert(length + buf_offset <= buffer_size);
			stamp::core::s_assert(length + pos <= buffer_v->size());

			std::copy(span, span + length, map_buffer_v + buf_offset);
			pos += length;
			if (pos >= buffer_v->size()) {
				eof_v = true;
			}
			return length;
		}
		virtual std::size_t do_try_get_span(char* span, std::size_t size) {
			eof_v = false;
			std::ptrdiff_t buf_offset = pos - buffer_pos;
			if (buf_offset < 0 || buf_offset >= buffer_size) return 0;
			std::ptrdiff_t length = std::min(
				(std::ptrdiff_t)size, 
				(std::ptrdiff_t)(buffer_size - buf_offset), 
				(std::ptrdiff_t)(buffer_v->size() - pos));
			stamp::core::s_assert(length + buf_offset <= buffer_size);
			stamp::core::s_assert(length + pos <= buffer_v->size());

			std::copy(map_buffer_v + buf_offset, map_buffer_v + buf_offset + length, span);
			pos += length;
			if (pos >= buffer_v->size()) {
				eof_v = true;
			}
			return length;
		}

		virtual bool do_await_ready() {
			if (pos < buffer_pos || pos > buffer_pos + buffer_size) {
				return false;
			}
			return true;
		}
		virtual bool do_await_suspend(std::coroutine_handle<> h) {
			std::size_t aligned_pos = (pos / buffer_size) * buffer_size;
			// TODO: copy current buffer (make sure to cut off at file end)
			// then copy new range into strand's buffer.
			buffer_pos = aligned_pos;
			return false; // resumes coroutine.
		}
		virtual bool do_await_resume() {
			return good();
		}
	public:
		template<typename ValueType>
		void set_pos(std::size_t index) {
			set_ppos(index * sizeof(ValueType));
			// set_gpos(index * sizeof(value_type)); // redundent, both set same pos.
		}
		template<typename ValueType>
		stamp::task::coroutine<ValueType> get_next() {
			ValueType v;
			co_await this->get_span(&v, sizeof(ValueType));
			co_return v;
		}
		template<typename ValueType>
		stamp::task::coroutine<ValueType> set_next(const_reference val) {
			ValueType v;
			co_await this->put_span(&v, sizeof(ValueType));
			co_return v;
		}
	};

	/* // example core for buffer_io_strand
	stamp::task::coroutine<void> run(buffer_iostream_strand<int>* strand) {
		strand->go_at(12);
		int v = co_await strand->get_next();
		co_await strand->set_next(v+3);
	}
	*/

	class buffer {
	public:
		using size_type = std::size_t;
	private:
		VkBuffer buffer_v;
		gptr_t raw_data_v;
	public:
		buffer();
		buffer(size_type size);
		// copy buffer in different memory type
		~buffer();
		buffer(const buffer&);
		buffer(buffer&&);
		buffer& operator =(const buffer&);
		buffer& operator =(buffer&&);

		// number of elements
		size_type size() const;
		size_type max_size() const;
		bool empty() const;
		VkBuffer handle() const;

		// awaiter
		auto clear() {

		}
		// awaiter
		auto resize(size_type new_size) {

		}

		buffer_io_strand co_iostream_strand() {
			return {this};
		}

		template<typename T>
		std::vector<T> to_vector() {

		}
		template<typename T>
		static buffer from_vector(const std::vector<T>&);
	};

	using image_storage_t = int;
	struct image_storage {
		enum : image_storage_t {
			rgba8
		};
	};

	class image2d {
	public:
		image2d(std::size_t width, std::size_t height, image_storage_t);
		~image2d();
		image2d(const image2d&) = delete;
		image2d(image2d&&) {

		}
		image2d& operator =(const image2d&) = delete;
		image2d& operator =(image2d&&) {

		}

		VkImage handle();

	};

	struct compute_shader_config {
		const std::string& file_name;
		int storage_buffer_count;
	};

	class compute_shader {
		VkDescriptorSetLayout desc_layout_v;
		VkPipelineLayout pipeline_layout_v;
		VkShaderModule shader_v;
		VkPipeline pipeline_v;
		VkDescriptorSet descriptor_v;
	public:
		compute_shader() {

		}
		compute_shader(const compute_shader_config& config) {

		}
		~compute_shader() {

		}

		compute_shader(const compute_shader&) = delete;
		compute_shader(compute_shader&&) {

		}
		compute_shader& operator =(const compute_shader&) = delete;
		compute_shader& operator =(compute_shader&&) {

		}
		
		template<typename T>
		void bind(int binding, const buffer&) {

		}
		void invoke(int x, int y, int z) {

		}
	};
}

#endif // GSPLAT_VULKAN_MEMORY_H