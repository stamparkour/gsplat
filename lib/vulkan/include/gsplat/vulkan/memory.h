#ifndef GSPLAT_VULKAN_MEMORY_H
#define GSPLAT_VULKAN_MEMORY_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <gsplat/core/read_file.h>
#include <filesystem>
#include <cstddef>
#include <utility>
#include <mutex>
//#include <stamp/task/stream.h>

// uses cpp coroutine/awaiters

namespace gsplat::vulkan {
	using gptr_t = void*;
	using command_buffer = void*;

	struct memory_type {
		enum : int {
			device_local,
			host_visible, // assumes highly volatile
		};
	};

	gptr_t malloc(std::size_t size, std::size_t align, memory_type);
	void free(gptr_t);

	class buffer;
	template<typename T>
	class buffer_iostream_strand : stamp::task::co_stream_strand {
	public:
		buffer_iostream_strand(buffer*);
		~buffer_iostream_strand();
	protected:
		virtual bool do_eof() const;
		virtual bool do_fail() const;
		virtual bool do_bad() const;

		virtual std::size_t do_get_ppos() const;
		virtual void do_set_ppos(std::size_t pos, seekdir);
		virtual std::size_t do_get_gpos() const;
		virtual void do_set_gpos(std::size_t pos, seekdir);

		virtual std::size_t do_try_put_span(const char* span, std::size_t size);
		virtual std::size_t do_try_get_span(char* span, std::size_t size);

		virtual bool do_await_ready();
		virtual bool do_await_suspend(std::coroutine_handle<> h);
		virtual std::size_t do_await_resume();
	};

	class buffer {
	public:
		using size_type = std::size_t;
	private:
	public:
		buffer();
		buffer(size_type size);
		// copy buffer in different memory type
		~buffer();
		buffer(const buffer&);
		buffer(buffer&&);
		buffer& operator =(const buffer&);
		buffer& operator =(buffer&&);

		size_type size() const;
		size_type max_size() const;
		bool empty() const;
		VkBuffer handle() const;

		// awaiter
		auto clear();
		// awaiter
		auto resize(size_type new_size);

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
		image2d(image2d&&) = delete;
		image2d& operator =(const image2d&) = delete;
		image2d& operator =(image2d&&) = delete;

		VkImage handle();
	};

    // Returns the path of the program executable since current working directory
    // may not be the actual path of the executable.
    std::filesystem::path executable_path() {
        // Cross-compatible code generated with help of claude. Some platforms untested.
        #ifdef _WIN32
            #include <windows.h>
            wchar_t buf[MAX_PATH];
            DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
            return std::filesystem::path(std::wstring(buf, n));
        #elif defined(__APPLE__)
            #include <mach-o/dyld.h>
            uint32_t size = 0;
            _NSGetExecutablePath(nullptr, &size);
            std::string buf(size, '\0');
            _NSGetExecutablePath(buf.data(), &size);
            return std::filesystem::canonical(buf.c_str());
        #else
            return std::filesystem::canonical("/proc/self/exe");
        #endif
    }

    struct compute_shader_config {
        const std::string& file_name;
        int storage_buffer_count;
        const std::string shader_dir = (executable_path().parent_path() / "shaders").string();
    };

    class compute_shader {
        VkDescriptorSetLayout desc_layout_v;
        VkPipelineLayout pipeline_layout_v;
        VkShaderModule shader_v;
        VkPipeline pipeline_v;
        VkDescriptorSet descriptor_v;
        VkFence fence_v;
        std::mutex invoke_mtx_v;
    public:
        compute_shader() {

        }
        compute_shader(const compute_shader_config& config) {
            // Create Descriptor Set Layout
            std::vector<VkDescriptorSetLayoutBinding> layout_bindings{};
            for (int i = 0; i < config.storage_buffer_count; i++) {
                layout_bindings.emplace_back(
                    i, // binding
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    1, // descriptor count
                    VK_SHADER_STAGE_COMPUTE_BIT, // stage flags
                    nullptr // immutable samplers
                );
            }
            VkDescriptorSetLayoutCreateInfo layout_ci = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .bindingCount = (uint32_t)layout_bindings.size(),
                .pBindings = layout_bindings.data()
            };
            if (VkResult r = vkCreateDescriptorSetLayout(vulkan_context::device(), &layout_ci, nullptr, &desc_layout_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not create descriptor set layout for computer shader. Vk Code: " + std::to_string(r));
            }

            // Load shader module file
            std::vector<char> shader_file = gsplat::core::load_entire_file_binary(config.shader_dir + "/" + config.file_name);

            // Create Shader Module from file
            VkShaderModuleCreateInfo mod_ci = {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .codeSize = shader_file.size(),
                .pCode = reinterpret_cast<const uint32_t*>(shader_file.data())
            };
            
            if (VkResult r = vkCreateShaderModule(vulkan_context::device(), &mod_ci, nullptr, &shader_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not create the shader module for file: " + config.file_name + "(Vk Code: " + std::to_string(r) + ")");
            }

            // Create Shader Pipeline Layout
            VkPipelineShaderStageCreateInfo stage_ci = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_v,
                .pName = "main",
                .pSpecializationInfo = nullptr
            };
            
            VkPipelineLayoutCreateInfo pipeline_layout_ci = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .setLayoutCount = 1,
                .pSetLayouts = &desc_layout_v,
                .pushConstantRangeCount = 0,
                .pPushConstantRanges = nullptr
            };

            if (VkResult r = vkCreatePipelineLayout(vulkan_context::device(), &pipeline_layout_ci, nullptr, &pipeline_layout_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not create the shader pipeline layout for file: " + config.file_name + "(Vk Code: " + std::to_string(r) + ")");
            }

            // Create Shader Pipeline
            VkComputePipelineCreateInfo pipeline_ci = {
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = stage_ci,
                .layout = pipeline_layout_v,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = 0
            };

            if (VkResult r = vkCreateComputePipelines(vulkan_context::device(), VK_NULL_HANDLE, 1, & pipeline_ci, nullptr, &pipeline_v);
                r < VK_SUCCESS
            ) {
                throw std::runtime_error("Could not create the shader pipeline for file: " + config.file_name + "(Vk Code: " + std::to_string(r) + ")");
            }

            VkDescriptorSetAllocateInfo desc_set_ai = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .pNext = nullptr,
                .descriptorPool = vulkan_context::descriptor_pool(),
                .descriptorSetCount = 1,
                .pSetLayouts = &desc_layout_v
            };
            if (VkResult r = vkAllocateDescriptorSets(vulkan_context::device(), &desc_set_ai, &descriptor_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not allocate descriptor sets. Vk Code: " + std::to_string(r));
            }

            VkFenceCreateInfo fence_ci = {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0
            };

            VkFence fence;
            if (VkResult r = vkCreateFence(vulkan_context::device(), &fence_ci, nullptr, &fence_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not create fence for shader pipeline. Vk Code: " + std::to_string(r));
            }

        }
        ~compute_shader() {
            destroy();
        }

        compute_shader(const compute_shader&) = delete;
        compute_shader& operator =(const compute_shader&) = delete;
        compute_shader(compute_shader&& o) noexcept
            : desc_layout_v(std::exchange(o.desc_layout_v, VK_NULL_HANDLE)),
              pipeline_layout_v(std::exchange(o.pipeline_layout_v, VK_NULL_HANDLE)),
              shader_v(std::exchange(o.shader_v, VK_NULL_HANDLE)),
              pipeline_v(std::exchange(o.pipeline_v, VK_NULL_HANDLE)),
              descriptor_v(std::exchange(o.descriptor_v, VK_NULL_HANDLE)),
              fence_v(std::exchange(o.fence_v, VK_NULL_HANDLE)) {}
        compute_shader& operator =(compute_shader&& o) noexcept {
            if (this != &o) {
                destroy();
                desc_layout_v = std::exchange(o.desc_layout_v, VK_NULL_HANDLE);
                pipeline_layout_v = std::exchange(o.pipeline_layout_v, VK_NULL_HANDLE);
                shader_v = std::exchange(o.shader_v, VK_NULL_HANDLE);
                pipeline_v = std::exchange(o.pipeline_v, VK_NULL_HANDLE);
                descriptor_v = std::exchange(o.descriptor_v, VK_NULL_HANDLE);
                fence_v = std::exchange(o.fence_v, VK_NULL_HANDLE);
            }
            return *this;
        }

        void destroy() noexcept {
            if (fence_v) {
                vkDestroyFence(vulkan_context::device(), fence_v, nullptr);
                fence_v = VK_NULL_HANDLE;
            }
            if (descriptor_v) {
                vkFreeDescriptorSets(vulkan_context::device(), vulkan_context::descriptor_pool(), 1, &descriptor_v);
                descriptor_v = VK_NULL_HANDLE;
            }
            if (pipeline_v) {
                vkDestroyPipeline(vulkan_context::device(), pipeline_v, nullptr);
                pipeline_v = VK_NULL_HANDLE;
            }
            if (pipeline_layout_v) {
                vkDestroyPipelineLayout(vulkan_context::device(), pipeline_layout_v, nullptr);
                pipeline_layout_v = VK_NULL_HANDLE;
            }
            if (shader_v) {
                vkDestroyShaderModule(vulkan_context::device(), shader_v, nullptr);
                shader_v = VK_NULL_HANDLE;
            }
            if (desc_layout_v) {
                vkDestroyDescriptorSetLayout(vulkan_context::device(), desc_layout_v, nullptr);
                desc_layout_v = VK_NULL_HANDLE;
            }
        }

        void bind(int binding, const buffer& b) {
            VkDescriptorBufferInfo buf_info = {
                .buffer = b.handle(),
                .offset = 0,
                .range = b.size()
            };
            VkWriteDescriptorSet buf_write = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = descriptor_v, // shaders' desc set
                .dstBinding = (uint32_t)binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &buf_info,
                .pTexelBufferView = nullptr
            };
            vkUpdateDescriptorSets(vulkan_context::device(), 1, &buf_write, 0, nullptr);
        }
        void invoke(int x, int y, int z) {
            std::lock_guard lock(invoke_mtx_v);
            VkCommandBufferAllocateInfo cmd_ai = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = vulkan_context::command_pool(),
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1
            };

            VkCommandBuffer cmd_buf;
            if (VkResult r = vkAllocateCommandBuffers(vulkan_context::device(), &cmd_ai, &cmd_buf);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not allocate command buffer for shader pipeline. Vk Code: " + std::to_string(r));
            }

            VkCommandBufferBeginInfo cmd_begin = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = 0,
                .pInheritanceInfo = nullptr
            };

            if (VkResult r = vkBeginCommandBuffer(cmd_buf, &cmd_begin); r != VK_SUCCESS) {
                throw std::runtime_error("Could not begin recording command buffer for shader pipeline. Vk Code: " + std::to_string(r));
            }

            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_v);
            vkCmdBindDescriptorSets(
                cmd_buf,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline_layout_v,
                0, 1,
                &descriptor_v,
                0, 0
            );
            vkCmdDispatch(cmd_buf, x, y, z);
            if (VkResult r = vkEndCommandBuffer(cmd_buf); r != VK_SUCCESS) {
                throw std::runtime_error("Could not end command buffer for shader pipeline. Vk Code: " + std::to_string(r));
            }

            VkSubmitInfo submit_info = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &cmd_buf,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr
            };

            if (VkResult r = vkResetFences(vulkan_context::device(), 1, &fence_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not reset fence for shader pipeline. Vk Code: " + std::to_string(r));
            }

            if (VkResult r = vkQueueSubmit(vulkan_context::queue(), 1, &submit_info, fence_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not submit queue for shader pipeline. Vk Code: " + std::to_string(r));
            }

            if (VkResult r = vkWaitForFences(vulkan_context::device(), 1, &fence_v, VK_TRUE, UINT64_MAX);
                r < VK_SUCCESS
            ) {
                throw std::runtime_error("Error waiting for shader pipeline to finish. Vk Code: " + std::to_string(r));
            }
            vkFreeCommandBuffers(vulkan_context::device(), vulkan_context::command_pool(), 1, &cmd_buf);
        }
    };
}

#endif // GSPLAT_VULKAN_MEMORY_H
