#ifndef GSPLAT_TRAIN_SHADER_STORE_H
#define GSPLAT_TRAIN_SHADER_STORE_H

#include <gsplat/vulkan/shader.h>
#include <gsplat/data/database.h>
#include <gsplat/core/read_file.h>
#include <string>

namespace gsplat::train {
	class shader_store {
		gsplat::data::database* database_v;
		gsplat::vulkan::shader calc_covariance_s;
		gsplat::vulkan::compute_shader_pipeline calc_covariance_sp;
	public:
		shader_store(const std::string& shader_dir, gsplat::data::database* db, gsplat::vulkan::command_queue* q) {
			this->database_v = db;

			std::vector<char> vec;
			
			vec = gsplat::core::load_entire_file_binary(shader_dir + "/calc_covariance.slang.spv");

			calc_covariance_s = gsplat::vulkan::shader{
				gsplat::vulkan::shader_creation_settings{
					.file_ptr = vec.data(),
					.file_size = vec.size(),
					.device = q->logical_device().get()
				}
			};
			calc_covariance_sp = gsplat::vulkan::compute_shader_pipeline{
				gsplat::vulkan::compute_shader_pipeline_settings{
					.storage_buffer_count = 2,
					.queue = q,
					.shader = &calc_covariance_s
				}
			};
		}

		void calc_covariance() {
			using namespace gsplat::vulkan;

			struct settings_t {
				int buffer_io_size;
			};

			array_buffer<settings_t> buf{buffer_creation_settings{
				.usage_transfer_src = true,
				.usage_storage_buffer = true,
				.size = 1,
				.memory = {
					device_memory_settings{
						.memory_host_visible = true,
						.memory_host_coherent = true,
					}
				},
				.queues = {
					calc_covariance_sp.queue()
				}
			}};
			{
				auto l = buf.memory_lock();
				l.data()->buffer_io_size = (int)database_v->gaussian_set().data().size();

			}
			calc_covariance_sp.bind(0, &(database_v->gaussian_set().buffer1_vulkan()));
			calc_covariance_sp.bind(1, &buf);
			calc_covariance_sp.invoke_compute(((int)database_v->gaussian_set().size()+1023)/1024, 1, 1);
		}
	};
}

#endif // GSPLAT_TRAIN_SHADER_STORE_H