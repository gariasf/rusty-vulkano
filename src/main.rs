use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::{sync, VulkanLibrary};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::format::{ClearColorValue, Format};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
                        PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::sync::GpuFuture;

mod compute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            // Buffer descriptor
            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                    uint index = gl_GlobalInvocationID.x;
                    buf.data[index] *= 12;
            }
        ",
    }
}

fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance = Instance::new(library, InstanceCreateInfo::default())
        .expect("failed to create instance");

    // Getting a device and its memory allocator
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("no devices available");

    let queue_family_index = physical_device.queue_family_properties()
        .iter().enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        })
        .expect("Couldn't find graphical queue family") as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    ).expect("failed to create device");

    let queue = queues.next().unwrap();
    let memory_allocator =
        Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    // todo!() Pass this to all examples to reduce duplication.
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    example_cpu_to_gpu_buffer_copy(queue_family_index, &device, &queue, &memory_allocator);

    example_storage_buffer_compute_shader(&device, &queue, &memory_allocator.clone());

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            // UNORM -> Unsigned normalized, normalized means that the color value in memory is
            // interpreted as a floating point number.
            // 0 will be interpreted as 0.0 and 255 will be interpreted as 1.0
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    ).unwrap();


    let image_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        // Image size is 1024 x 1024, and each of the four (RGBA) color values for each
        // pixel have 8 bits
        (0..1024 * 1024 * 4).map(|_| 0u8)
    ).expect("failed to create buffer");

    let mut command_builder =
        AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();


    // Clearing an image means to fill the image buffer
    // with a color.
    command_builder.clear_color_image(
        ClearColorImageInfo {
            // We use 0.0 and 1.0 here because of the image format
            // 1.0 will represent 255.
            clear_value: ClearColorValue::Float([0.0, 0.0, 1.0, 1.0]),
            ..ClearColorImageInfo::image(image.clone())
        }
    ).unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            image_buffer.clone(),
        )).unwrap();

    let command_buffer = command_builder.build().unwrap();

    let image_buffer_future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    image_buffer_future.wait(None).unwrap();

    let image_buffer_content = image_buffer.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &image_buffer_content[..]).unwrap();
    image.save("image.png").unwrap();
    println!("Image clearing and saving succeeded!");
}

fn example_storage_buffer_compute_shader(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
) {
// Buffer using storage
    let data_iter = 0..65536u32;
    let data_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data_iter,
    )
        .expect("failed to create buffer");

    // Handle compute shader pipeline
    let shader = compute_shader::load(device.clone())
        .expect("failed to create shader module");

    let compute_shader = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(compute_shader);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    ).unwrap();

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    ).expect("failed to create compute pipeline");

    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
        [],
    ).unwrap();

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut command_buffer_builder =
        AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

    // We want to use 1024 work groups in a single dimension (data is single-dimsensional)
    let work_group_counts = [1024, 1, 1];

    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )
        .unwrap()
        .dispatch(work_group_counts)
        .unwrap();

    let command_buffer = command_buffer_builder.build().unwrap();

    let command_buffer_future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    command_buffer_future.wait(None).unwrap();

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    println!("Send compute shader command succeeded!");
}

fn example_cpu_to_gpu_buffer_copy(
    queue_family_index: u32,
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
) {
// Create source (CPU) and destination (GPU) buffers
    let source_content: Vec<i32> = (0..64).collect();
    let source_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        source_content,
    )
        .expect("failed to create source buffer");

    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let destination_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        destination_content,
    )
        .expect("failed to create destination buffer");

    // Prepare and send copy command
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut command_buffer_builder =
        AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

    command_buffer_builder.copy_buffer(CopyBufferInfo::buffers(
        source_buffer.clone(),
        destination_buffer.clone(),
    )).unwrap();

    let command_buffer = command_buffer_builder.build().unwrap();

    let command_future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    command_future.wait(None).unwrap();

    // Check result
    let src_content = source_buffer.read().unwrap();
    let destination_content = destination_buffer.read().unwrap();
    assert_eq!(&*src_content, &*destination_content);

    println!("Send copy buffer command succeeded!");
}
