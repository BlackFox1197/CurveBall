#![feature(rust_2018_preview, use_extern_macros, nll)]

mod icosphere;
mod shaders;
mod vertex;

use std::mem;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cgmath::{Deg, Euler, Matrix4, Point3, Quaternion, Rad, SquareMatrix, Vector3};
use log::*;
use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::instance::debug::{DebugCallback, MessageTypes};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::{viewport::Viewport, GraphicsPipeline};
use vulkano::swapchain::{
    self, AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError,
};
use vulkano::sync::{now, FlushError, GpuFuture};
use vulkano::{ordered_passes_renderpass, single_pass_renderpass};
use vulkano_win::VkSurfaceBuild;
use winit::{
    DeviceEvent, ElementState, Event, EventsLoop, KeyboardInput, VirtualKeyCode, WindowBuilder,
    WindowEvent,
};

use crate::icosphere::icosphere;

// TODO: mesh optimization
// - http://gfx.cs.princeton.edu/pubs/Sander_2007_%3ETR/tipsy.pdf
// - https://tomforsyth1000.github.io/papers/fast_vert_cache_opt.html
// - http://www.martin.st/thesis/
// - https://github.com/zeux/meshoptimizer

fn main() {
    env_logger::init();

    let debug = true;
    let instance = {
        // All the window-drawing functionalities are part of non-core extensions that we need
        // to enable manually. To do so, we ask the `vulkano_win` crate for the list of extensions
        // required to draw to a window.
        let mut extensions = vulkano_win::required_extensions();
        extensions.ext_debug_report = debug;
        let layers = if debug {
            vec!["VK_LAYER_LUNARG_standard_validation"]
        } else {
            Vec::new()
        };
        Instance::new(None, &extensions, &layers).expect("failed to create Vulkan instance")
    };

    // Must be kept alive or the mssages will disappear!
    let debug_callback = if debug {
        let all_message_types = MessageTypes {
            error: true,
            warning: true,
            performance_warning: true,
            information: true,
            debug: true,
        };
        Some(
            DebugCallback::new(&instance, all_message_types, |msg| {
                macro_rules! fmt {
                    () => {
                        "[VK,{}] {}"
                    };
                }
                if msg.ty.error {
                    error!(fmt!(), msg.layer_prefix, msg.description);
                } else if msg.ty.warning {
                    warn!(fmt!(), msg.layer_prefix, msg.description);
                } else if msg.ty.performance_warning {
                    warn!(fmt!(), msg.layer_prefix, msg.description);
                } else if msg.ty.information {
                    info!(fmt!(), msg.layer_prefix, msg.description);
                } else if msg.ty.debug {
                    debug!(fmt!(), msg.layer_prefix, msg.description);
                } else {
                    unreachable!("unknown debug message type")
                };
            }).unwrap(),
        )
    } else {
        None
    };

    // We then choose which physical device to use.
    //
    // In a real application, there are three things to take into consideration:
    //
    // - Some devices may not support some of the optional features that may be required by your
    //   application. You should filter out the devices that don't support your app.
    //
    // - Not all devices can draw to a certain surface. Once you create your window, you have to
    //   choose a device that is capable of drawing to it.
    //
    // - You probably want to leave the choice between the remaining devices to the user.
    //
    // For the sake of the example we are just going to use the first device, which should work
    // most of the time.
    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );

    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();

    surface.window().grab_cursor(true).unwrap();
    surface.window().hide_cursor(true);

    // In a real-life application, we would probably use at least a graphics queue and a transfers
    // queue to handle data transfers in parallel. In this example we only use one queue.
    let queue_family = physical
        .queue_families()
        .find(|&q| {
            // We take the first queue_family that supports drawing to our window.
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        })
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        Device::new(
            physical,
            physical.supported_features(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        ).expect("failed to create device")
    };

    // We only requested one queue.
    let queue = queues.next().unwrap();

    // The dimensions of the surface.
    // This variable needs to be mutable since the viewport can change size.
    let mut dimensions;

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating
    // a swapchain allocates the color buffers that will contain the image that will ultimately
    // be visible on the screen. These images are returned alongside with the swapchain.
    let (mut swapchain, mut images) = {
        let caps = surface
            .capabilities(physical)
            .expect("failed to get surface capabilities");

        dimensions = caps.current_extent.unwrap_or([1024, 768]);

        // Only determines how the alpha value of the final window pixels are interpreted.
        // (opaque vs. transparent window)
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        println!("Image format: {:?}", format);

        let present_mode = if caps.present_modes.mailbox {
            PresentMode::Mailbox
        } else {
            PresentMode::Fifo
        };

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count + 1, // TODO: What is the correct number for triple buffering?
            format,
            dimensions,
            1, // layers; multiple needed for 3D
            caps.supported_usage_flags,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            present_mode,
            true, // clipped
            None, // old_swapchain
        ).expect("failed to create swapchain")
    };

    let (vertices, indices) = icosphere(6);
    let vertex_buffer = {
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), vertices.iter().cloned())
            .expect("failed to create buffer")
    };
    let index_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), indices.iter().cloned())
            .expect("failed to create buffer");

    let vs = shaders::vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = shaders::fs::Shader::load(device.clone()).expect("failed to create shader module");

    // TODO: pull out into a `Camera`; also handle swapchain recreations
    let mut proj = cgmath::perspective(
        Rad(std::f32::consts::FRAC_PI_2),
        { dimensions[0] as f32 / dimensions[1] as f32 },
        0.01,
        100.0,
    );
    let mut look_at_dir;
    let mut pos = Point3::new(0.0, 0.0, -3.0);
    let mut view;

    let mut delta = Vector3::new(0.0, 0.0, 0.0);
    let mut rotation = Quaternion::from(Euler {
        x: Deg(0.0),
        y: Deg(0.0),
        z: Deg(0.0),
    });

    let uniform_buffer =
        CpuBufferPool::<shaders::vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let render_pass = Arc::new(
        single_pass_renderpass!(device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1, // TODO: Figure out if MSAA is possible atm
                }
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {}
            }
        ).unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
        // We need to indicate the layout of the vertices.
        // The type `SingleBufferDefinition` actually contains a template parameter corresponding
        // to the type of each vertex. But in this code it is automatically inferred.
        .vertex_input_single_buffer()
        // A Vulkan shader can in theory contain multiple entry points, so we have to specify
        // which one. The `main` word of `main_entry_point` actually corresponds to the name of
        // the entry point.
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
        // .polygon_mode_line()
        .front_face_counter_clockwise()
        .cull_mode_back()
        // Use a resizable viewport set to draw over the entire window
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        // We have to indicate which subpass of which render pass this pipeline is going to be used
        // in. The pipeline will only be usable from this particular subpass.
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap(),
    );

    // The render pass we created above only describes the layout of our framebuffers. Before we
    // can draw we also need to create the actual framebuffers.
    //
    // Since we need to draw to multiple images, we are going to create a different framebuffer for
    // each image.
    let mut framebuffers: Option<Vec<Arc<Framebuffer<_, _>>>> = None;

    // Initialization is finally finished!

    // In some situations, the swapchain will become invalid by itself. This includes for example
    // when the window is resized (as the images of the swapchain will no longer match the
    // window's) or, on Android, when the application went to the background and goes back to the
    // foreground.
    //
    // In this situation, acquiring a swapchain image or presenting it will return an error.
    // Rendering to an image of that swapchain will not produce any error, but may or may not work.
    // To continue rendering, we need to recreate the swapchain by creating a new swapchain.
    // Here, we remember that we need to do this for the next loop iteration.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Box::new(now(device.clone())) as Box<GpuFuture>;

    let mut last_sec = Instant::now();
    let mut fps = 0;

    loop {
        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU has
        // already processed, and frees the resources that are no longer needed.
        previous_frame_end.cleanup_finished();

        let now = Instant::now();
        if now - last_sec >= Duration::from_secs(1) {
            println!("fps: {}", fps);
            last_sec = now;
            fps = 0;
        } else {
            fps += 1;
        }

        // If the swapchain needs to be recreated, recreate it
        if recreate_swapchain {
            // Get the new dimensions for the viewport/framebuffers.
            dimensions = surface
                .capabilities(physical)
                .expect("failed to get surface capabilities")
                .current_extent
                .unwrap();

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.
                // Simply restarting the loop is the easiest way to fix this issue.
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

            proj = cgmath::perspective(
                Rad(std::f32::consts::FRAC_PI_2),
                { dimensions[0] as f32 / dimensions[1] as f32 },
                0.01,
                100.0,
            );

            mem::replace(&mut swapchain, new_swapchain);
            mem::replace(&mut images, new_images);

            framebuffers = None;

            recreate_swapchain = false;
        }

        // Because framebuffers contains an Arc on the old swapchain, we need to
        // recreate framebuffers as well.
        if framebuffers.is_none() {
            let new_framebuffers = Some(
                images
                    .iter()
                    .map(|image| {
                        Arc::new(
                            Framebuffer::start(render_pass.clone())
                                .add(image.clone())
                                .unwrap()
                                .build()
                                .unwrap(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            mem::replace(&mut framebuffers, new_framebuffers);
        }

        look_at_dir = rotation * Vector3::new(0.0, 0.0, 1.0);
        let up = rotation * Vector3::new(0.0, -1.0, 0.0);
        pos += rotation * delta;
        view = Matrix4::look_at_dir(pos, look_at_dir, up);

        let uniform_buffer_subbuffer = {
            let uniform_data = shaders::vs::ty::Data {
                world: Matrix4::identity().into(),
                view: view.into(),
                proj: proj.into(),
            };

            uniform_buffer.next(uniform_data).unwrap()
        };

        let set = Arc::new(
            PersistentDescriptorSet::start(pipeline.clone(), 0)
                .add_buffer(uniform_buffer_subbuffer)
                .unwrap()
                .build()
                .unwrap(),
        );

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            // Before we can draw, we have to *enter a render pass*. There are two methods to do
            // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
            // not covered here.
            //
            // The third parameter builds the list of values to clear the attachments with. The API
            // is similar to the list of attachments when building the framebuffers, except that
            // only the attachments that use `load: Clear` appear in the list.
            .begin_render_pass(framebuffers.as_ref().unwrap()[image_num].clone(), false,
                               vec![[0.1, 0.1, 0.1, 1.0].into()])
            .unwrap()

            // We are now inside the first subpass of the render pass. We add a draw command.
            //
            // The last two parameters contain the list of resources to pass to the shaders.
            // Since we used an `EmptyPipeline` object, the objects have to be `()`.
            .draw_indexed(pipeline.clone(),
                  DynamicState {
                      line_width: None,
                      // TODO: Find a way to do this without having to dynamically allocate a Vec every frame.
                      viewports: Some(vec![Viewport {
                          origin: [0.0, 0.0],
                          dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                          depth_range: 0.0 .. 1.0,
                      }]),
                      scissors: None,
                  },
                  vertex_buffer.clone(), index_buffer.clone(), set.clone(), ())
            .unwrap()

            // We leave the render pass by calling `draw_end`. Note that if we had multiple
            // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
            // next subpass.
            .end_render_pass()
            .unwrap()
            .build().unwrap();

        let future = previous_frame_end
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
        }

        // Note that in more complex programs it is likely that one of `acquire_next_image`,
        // `command_buffer::submit`, or `present` will block for some time. This happens when the
        // GPU's queue is full and the driver has to wait until the GPU finished some work.
        //
        // Unfortunately the Vulkan API doesn't provide any way to not wait or to detect when a
        // wait would happen. Blocking may be the desired behavior, but if you don't want to
        // block you should spawn a separate thread dedicated to submissions.

        let mut done = false;
        events_loop.poll_events(|ev| match ev {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => done = true,
            Event::WindowEvent {
                event: WindowEvent::Focused(b),
                ..
            } => {
                surface.window().hide_cursor(b);
            }
            Event::DeviceEvent {
                event:
                    DeviceEvent::Key(KeyboardInput {
                        virtual_keycode: Some(key_code),
                        state,
                        ..
                    }),
                ..
            } => {
                let d = match key_code {
                    VirtualKeyCode::W => Vector3::new(0.0, 0.0, 0.05),
                    VirtualKeyCode::A => Vector3::new(-0.05, 0.0, 0.0),
                    VirtualKeyCode::S => Vector3::new(0.0, 0.0, -0.05),
                    VirtualKeyCode::D => Vector3::new(0.05, 0.0, 0.0),
                    VirtualKeyCode::E => Vector3::new(0.0, 0.05, 0.0),
                    VirtualKeyCode::Q => Vector3::new(0.0, -0.05, 0.0),
                    _ => return,
                };
                match state {
                    ElementState::Pressed => delta += d,
                    ElementState::Released => delta -= d,
                }
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                let d = Quaternion::from(Euler {
                    x: Deg(0.1 * delta.1 as f32),
                    y: Deg(0.1 * delta.0 as f32),
                    z: Deg(0.0),
                });
                rotation = rotation * d;
                // let euler = Euler::from(rotation);
                // d = Quaternion::from(Euler {
                //     x: Deg(0.0),
                //     y: Deg(0.0),
                //     z: (-euler.z).into(),
                // });
                // rotation = rotation * d;
            }
            e => warn!("{:?}", e),
        });
        if done {
            // Keep debug_callback alive until here
            std::mem::drop(debug_callback);
            return;
        }
    }
}
