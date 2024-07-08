import shaderCode from "@/shaders/shader.wgsl";
import mipmapShaderCode from "@/shaders/mipmap.wgsl";
import { mat4, utils, vec3 } from "wgpu-matrix";
import { GUI, GUIController } from "dat.gui";
import {
  loadImageBitmap,
  calculateMipLevelCount,
  RollingAverage,
} from "@/utils";

const fpsAvg = new RollingAverage();
const cpuTimeAvg = new RollingAverage();
const gpuTimeAvg = new RollingAverage();

export default class Renderer {
  private adapter: GPUAdapter;
  private device: GPUDevice;
  private hasTimestamp: boolean;
  private querySet: GPUQuerySet;
  private resolveBuffer: GPUBuffer;
  private resultBuffer: GPUBuffer;
  private canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private depthTexture: GPUTexture;
  private depthTextureView: GPUTextureView;
  private presentationFormat: GPUTextureFormat;
  private renderPipeline: GPURenderPipeline;
  private vertexBuffer: GPUBuffer;
  private indexBuffer: GPUBuffer;
  private bindGroups: Array<Array<GPUBindGroup>>;
  private uniformBuffer: GPUBuffer;

  private current: number;

  private fpsController: GUIController;
  private cpuTimeController: GUIController;
  private gpuTimeController: GUIController;
  private fovYController: GUIController;
  private xController: GUIController;
  private yController: GUIController;
  private zController: GUIController;
  private yawController: GUIController;
  private mipmapsController: GUIController;
  private addressModeUController: GUIController;
  private addressModeVController: GUIController;
  private magFilterController: GUIController;
  private minFilterController: GUIController;

  public constructor() {
    this.bindGroups = new Array<Array<GPUBindGroup>>(
      new Array<GPUBindGroup>(),
      new Array<GPUBindGroup>(),
      new Array<GPUBindGroup>()
    );

    this.current = 0;

    this.drawFrame = this.drawFrame.bind(this);
  }

  public async render() {
    await this.init();
    this.run();
  }

  private async init() {
    this.checkWebGPUSupport();
    await this.requestAdapter();
    await this.requestDevice();
    this.getCanvas();
    this.configContext();
    this.createDepthTexture();
    this.createRenderPipeline();
    this.createVertexBuffer();
    this.createIndexBuffer();
    this.createUniformBuffer();
    await this.createTexture();
    this.initGUI();
  }

  private checkWebGPUSupport() {
    if (!navigator.gpu) {
      throw Error("WebGPU not supported.");
    }
  }

  private async requestAdapter() {
    this.adapter = await navigator.gpu?.requestAdapter();
    if (!this.adapter) {
      throw Error("Failed to request WebGPU adapter.");
    }
  }

  private async requestDevice() {
    this.hasTimestamp = this.adapter?.features.has("timestamp-query");
    const requiredFeatures: GPUFeatureName[] = this.hasTimestamp
      ? ["timestamp-query"]
      : [];

    this.device = await this.adapter?.requestDevice({
      label: `GPU Device ${
        this.hasTimestamp && "with feature: timestamp-query"
      }`,
      requiredFeatures,
    });
    if (!this.device) {
      throw Error("Failed to request WebGPU device.");
    }

    if (this.hasTimestamp) {
      this.querySet = this.device.createQuerySet({
        label: "GPU Query Set with Type: timestamp",
        type: "timestamp",
        count: 2,
      });

      this.resolveBuffer = this.device.createBuffer({
        label: "GPU Buffer: Resolve",
        size: this.querySet.count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });

      this.resultBuffer = this.device.createBuffer({
        label: "GPU Buffer: Result",
        size: this.resolveBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    this.device.lost.then((info) => {
      if (info.reason !== "destroyed") {
        throw new Error(`WebGPU device was lost: ${info.message}`);
      }
    });
  }

  private getCanvas() {
    this.canvas = document.querySelector("canvas");
    if (!this.canvas) {
      throw Error("Failed to find canvas element.");
    }

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const canvas = entry.target as HTMLCanvasElement;
        const width = entry.contentBoxSize[0].inlineSize;
        const height = entry.contentBoxSize[0].blockSize;
        canvas.width = Math.max(
          1,
          Math.min(width, this.device.limits.maxTextureDimension2D)
        );
        canvas.height = Math.max(
          1,
          Math.min(height, this.device.limits.maxTextureDimension2D)
        );
        this.depthTexture.destroy();
        this.createDepthTexture();
      }
    });
    observer.observe(this.canvas);
  }

  private configContext() {
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) {
      throw Error("Failed to get WebGPU context from canvas.");
    }

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: "premultiplied",
    });
  }

  private async createDepthTexture() {
    this.depthTexture = this.device.createTexture({
      label: "GPU Texture: Depth Texture",
      size: [this.canvas.width, this.canvas.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.depthTextureView = this.depthTexture.createView({
      label: "GPU Texture View: Depth Texture View",
    });
  }

  private createRenderPipeline() {
    const shaderModule = this.createShaderModule(
      "GPU Shader Module",
      shaderCode
    );

    this.renderPipeline = this.device.createRenderPipeline({
      label: "GPU Render Pipeline",
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: (4 + 2) * Float32Array.BYTES_PER_ELEMENT,
            stepMode: "vertex" as GPUVertexStepMode,
            attributes: [
              {
                format: "float32x4" as GPUVertexFormat,
                offset: 0,
                shaderLocation: 0,
              },
              {
                format: "float32x2" as GPUVertexFormat,
                offset: 4 * Float32Array.BYTES_PER_ELEMENT,
                shaderLocation: 1,
              },
            ],
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "back",
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [
          {
            format: this.presentationFormat,
          },
        ],
      },
    });
  }

  private createVertexBuffer() {
    // prettier-ignore
    const vertices = new Float32Array([
      -1.0,  1.0,  1.0, 1.0, 0.0, 0.0, 
      -1.0,  1.0,  1.0, 1.0, 1.0, 0.0, 
      -1.0,  1.0,  1.0, 1.0, 0.0, 1.0, 
      -1.0, -1.0,  1.0, 1.0, 0.0, 1.0, 
      -1.0, -1.0,  1.0, 1.0, 1.0, 1.0, 
      -1.0, -1.0,  1.0, 1.0, 0.0, 0.0, 
       1.0,  1.0,  1.0, 1.0, 1.0, 0.0,
       1.0,  1.0,  1.0, 1.0, 0.0, 0.0, 
       1.0,  1.0,  1.0, 1.0, 1.0, 1.0, 
       1.0, -1.0,  1.0, 1.0, 1.0, 1.0, 
       1.0, -1.0,  1.0, 1.0, 0.0, 1.0, 
       1.0, -1.0,  1.0, 1.0, 1.0, 0.0, 
      -1.0,  1.0, -1.0, 1.0, 1.0, 0.0, 
      -1.0,  1.0, -1.0, 1.0, 0.0, 0.0,
      -1.0,  1.0, -1.0, 1.0, 0.0, 0.0, 
      -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 
      -1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 
      -1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 
       1.0,  1.0, -1.0, 1.0, 0.0, 0.0, 
       1.0,  1.0, -1.0, 1.0, 1.0, 0.0, 
       1.0,  1.0, -1.0, 1.0, 1.0, 0.0, 
       1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 
       1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
       1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
    ]);

    this.vertexBuffer = this.createBuffer(
      "GPU Buffer: Vertex",
      vertices.byteLength,
      GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    );

    this.device.queue.writeBuffer(
      this.vertexBuffer,
      0,
      vertices.buffer,
      0,
      vertices.byteLength
    );
  }

  private createIndexBuffer() {
    // prettier-ignore
    const indices = new Uint32Array([
       0,  3,  6,  6,  3,  9,
       7, 10, 19, 19, 10, 22,
      18, 21, 12, 12, 21, 15,
      13, 16,  1,  1, 16,  4,
      14,  2, 20, 20,  2,  8,
       5, 17, 11, 11, 17, 23,
    ]);

    this.indexBuffer = this.createBuffer(
      "GPU Buffer: Index",
      indices.byteLength,
      GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    );

    this.device.queue.writeBuffer(
      this.indexBuffer,
      0,
      indices.buffer,
      0,
      indices.byteLength
    );
  }

  private createUniformBuffer() {
    this.uniformBuffer = this.createBuffer(
      `GPU Uniform Buffer: MVP Matrix`,
      4 * 4 * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    const bindGroup = this.device.createBindGroup({
      label: `GPU Bind Group 0: MVP Matrix`,
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.uniformBuffer,
          },
        },
      ],
    });
    this.bindGroups[0].push(bindGroup);
  }

  private async createTexture() {
    const contianerImageBitMap = await loadImageBitmap("images/container.jpg");
    const faceImageBitMap = await loadImageBitmap("images/awesomeface.png");

    for (let i = 0; i < 2; ++i) {
      const contianerMipLevelCount = i
        ? calculateMipLevelCount(
            contianerImageBitMap.width,
            contianerImageBitMap.height
          )
        : 1;

      const contianerTexture = this.createTextureFromSource(
        `GPU Texture: Contianer ${i}`,
        contianerImageBitMap,
        contianerMipLevelCount
      );

      const contianerTextureView = contianerTexture.createView({
        label: `GPU Texture View: Contianer ${i}`,
      });

      const faceMipLevelCount = i
        ? calculateMipLevelCount(faceImageBitMap.width, faceImageBitMap.height)
        : 1;

      const faceTexture = this.createTextureFromSource(
        `GPU Texture: Face ${i}`,
        faceImageBitMap,
        faceMipLevelCount
      );

      const faceTextureView = faceTexture.createView({
        label: `GPU Texture View: Face ${i}`,
      });

      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 1: Face ${i}`,
        layout: this.renderPipeline.getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: contianerTextureView,
          },
          {
            binding: 1,
            resource: faceTextureView,
          },
        ],
      });
      this.bindGroups[1].push(bindGroup);
    }

    for (let i = 0; i < 16; ++i) {
      const sampler = this.device.createSampler({
        label: `GPU Sampler: Sampler ${i}`,
        addressModeU: i & 1 ? "repeat" : "clamp-to-edge",
        addressModeV: i & 2 ? "repeat" : "clamp-to-edge",
        magFilter: i & 4 ? "linear" : "nearest",
        minFilter: i & 8 ? "linear" : "nearest",
      });

      const bindGroup = this.device.createBindGroup({
        label: `GPU Bind Group 2: Sampler ${i}`,
        layout: this.renderPipeline.getBindGroupLayout(2),
        entries: [
          {
            binding: 0,
            resource: sampler,
          },
        ],
      });
      this.bindGroups[2].push(bindGroup);
    }
  }

  private createShaderModule(label: string, code: string) {
    const shaderModule = this.device.createShaderModule({
      label,
      code,
    });

    return shaderModule;
  }

  private createBuffer(
    label: string,
    size: number,
    usage: GPUBufferUsageFlags
  ) {
    const buffer = this.device.createBuffer({
      label,
      size,
      usage,
    });

    return buffer;
  }

  private createTextureFromSource(
    label: string,
    source: GPUImageCopyExternalImageSource,
    mipLevelCount: number
  ) {
    let width: number;
    let height: number;
    if (source instanceof HTMLVideoElement) {
      width = source.videoWidth;
      height = source.videoHeight;
    } else if (source instanceof VideoFrame) {
      width = source.codedWidth;
      height = source.codedHeight;
    } else {
      width = source.width;
      height = source.height;
    }

    const texture = this.device.createTexture({
      label,
      format: "rgba8unorm",
      mipLevelCount,
      size: [width, height],
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.copySourceToTexture(this.device, source, texture);

    return texture;
  }

  private copySourceToTexture(
    device: GPUDevice,
    source: GPUImageCopyExternalImageSource,
    texture: GPUTexture
  ) {
    let width: number;
    let height: number;
    if (source instanceof HTMLVideoElement) {
      width = source.videoWidth;
      height = source.videoHeight;
    } else if (source instanceof VideoFrame) {
      width = source.codedWidth;
      height = source.codedHeight;
    } else {
      width = source.width;
      height = source.height;
    }

    device.queue.copyExternalImageToTexture(
      { source },
      { texture },
      { width, height }
    );

    if (texture.mipLevelCount > 1) {
      this.generateMipmaps()(device, texture);
    }
  }

  private generateMipmaps() {
    let sampler: GPUSampler;
    let module: GPUShaderModule;
    const pipelineByFormat = new Map<GPUTextureFormat, GPURenderPipeline>();

    const context = this;

    return function generateMipmaps(device: GPUDevice, texture: GPUTexture) {
      if (!module) {
        module = device.createShaderModule({
          label: "GPU Shader Module: Mipmap Generation",
          code: mipmapShaderCode,
        });
        sampler = device.createSampler({
          minFilter: "linear",
        });
      }

      if (!pipelineByFormat.has(texture.format)) {
        const pipeline = device.createRenderPipeline({
          label: "GPU Render Pipeline: Mipmap Generation ",
          layout: "auto",
          vertex: {
            module: module,
            entryPoint: "vs_main",
            buffers: [
              {
                arrayStride: (4 + 2) * Float32Array.BYTES_PER_ELEMENT,
                stepMode: "vertex" as GPUVertexStepMode,
                attributes: [
                  {
                    format: "float32x4" as GPUVertexFormat,
                    offset: 0,
                    shaderLocation: 0,
                  },
                  {
                    format: "float32x2" as GPUVertexFormat,
                    offset: 4 * Float32Array.BYTES_PER_ELEMENT,
                    shaderLocation: 1,
                  },
                ],
              },
            ],
          },
          primitive: {
            topology: "triangle-list",
          },
          fragment: {
            module,
            entryPoint: "fs_main",
            targets: [
              {
                format: texture.format,
              },
            ],
          },
        });
        pipelineByFormat.set(texture.format, pipeline);
      }

      const pipeline = pipelineByFormat.get(texture.format);

      // vertex buffer
      const data = [
        {
          vertex: [-1.0, 1.0, 0.0, 1.0],
          texCoord: [0.0, 0.0],
        },
        {
          vertex: [-1.0, -1.0, 0.0, 1.0],
          texCoord: [0.0, 1.0],
        },
        {
          vertex: [1.0, 1.0, 0.0, 1.0],
          texCoord: [1.0, 0.0],
        },
        {
          vertex: [1.0, -1.0, 0.0, 1.0],
          texCoord: [1.0, 1.0],
        },
      ];

      let vertexOffset = 0;
      let texCoordOffset = 4;
      const vertices = new Float32Array(4 * 6);
      for (let i = 0; i < 4; ++i) {
        vertices.set(data[i].vertex, vertexOffset);
        vertices.set(data[i].texCoord, texCoordOffset);
        vertexOffset += 6;
        texCoordOffset += 6;
      }

      const vertexBuffer = context.createBuffer(
        "GPU Buffer: Mipmaps Generation Vertex",
        vertices.byteLength,
        GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
      );

      device.queue.writeBuffer(
        vertexBuffer,
        0,
        vertices.buffer,
        0,
        vertices.byteLength
      );

      // index buffer
      const indices = new Uint32Array([0, 1, 2, 2, 1, 3]);

      const indexBuffer = context.createBuffer(
        "GPU Buffer: Mipmap Generation Index ",
        indices.byteLength,
        GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
      );

      device.queue.writeBuffer(
        indexBuffer,
        0,
        indices.buffer,
        0,
        indices.byteLength
      );

      const encoder = device.createCommandEncoder({
        label: "GPU Command Encoder: Mipmap Generation",
      });

      let width = texture.width;
      let height = texture.height;
      let baseMipLevel = 0;
      while (width > 1 || height > 1) {
        width = Math.max(1, width / 2);
        height = Math.max(1, height / 2);

        const bindGroup = device.createBindGroup({
          label: "GPU Bind Group: Mipmap Generation",
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            {
              binding: 0,
              resource: sampler,
            },
            {
              binding: 1,
              resource: texture.createView({ baseMipLevel, mipLevelCount: 1 }),
            },
          ],
        });

        ++baseMipLevel;

        const colorAttachments: GPURenderPassColorAttachment[] = [
          {
            view: texture.createView({ baseMipLevel, mipLevelCount: 1 }),
            loadOp: "clear",
            storeOp: "store",
          },
        ];

        const passDescriptor: GPURenderPassDescriptor = {
          label: "GPU Render Pass Descriptor: Mipmap Generation",
          colorAttachments,
        };

        const pass = encoder.beginRenderPass(passDescriptor);
        pass.setPipeline(pipeline);
        pass.setVertexBuffer(0, vertexBuffer);
        pass.setIndexBuffer(indexBuffer, "uint32");
        pass.setBindGroup(0, bindGroup);
        pass.drawIndexed(6);
        pass.end();
      }

      device.queue.submit([encoder.finish()]);
    };
  }

  private initGUI() {
    const gui = new GUI({
      name: "My GUI",
      autoPlace: true,
      hideable: true,
      width: 300,
    });

    // profiling GUI
    const profiler = { fps: "0", cpuTime: "0", gpuTime: "0" };

    const profilerGUI = gui.addFolder("Profiler");
    profilerGUI.closed = false;
    this.fpsController = profilerGUI
      .add({ display: profiler.fps }, "display")
      .name("FPS");
    this.cpuTimeController = profilerGUI
      .add({ display: profiler.cpuTime }, "display")
      .name("CPU Time (ms)");
    this.gpuTimeController = profilerGUI
      .add({ display: profiler.gpuTime }, "display")
      .name("GPU Time (µs)");

    // camera GUI
    const camera = {
      reset: () => {
        this.fovYController.setValue(45.0);
        this.xController.setValue(3.0);
        this.yController.setValue(3.0);
        this.zController.setValue(10.0);
        this.yawController.setValue(0.0);
      },
      fovY: 45.0,
      position: {
        x: 3.0,
        y: 3.0,
        z: 10.0,
      },
      rotation: {
        yaw: 0.0,
      },
    };
    const cameraGUI = gui.addFolder("Camera");
    cameraGUI.closed = false;
    cameraGUI.add(camera, "reset").name("Reset");
    this.fovYController = cameraGUI
      .add(camera, "fovY", 0.0, 180.0, 0.1)
      .name("FoV (Y)");
    // camera postion GUI
    const cameraPositionGUI = cameraGUI.addFolder("Position");
    cameraPositionGUI.closed = false;
    this.xController = cameraPositionGUI
      .add(camera.position, "x")
      .step(0.1)
      .name("X");
    this.yController = cameraPositionGUI
      .add(camera.position, "y")
      .step(0.1)
      .name("Y");
    this.zController = cameraPositionGUI
      .add(camera.position, "z")
      .step(0.1)
      .name("Z");
    // camera rotation GUI
    const cameraRotationGUI = cameraGUI.addFolder("Rotation");
    cameraRotationGUI.closed = false;
    this.yawController = cameraRotationGUI
      .add(camera.rotation, "yaw", -180.0, 180.0, 0.1)
      .step(0.1)
      .name("Yaw");

    // texture options GUI
    const textureOptions = {
      mimaps: true,
      addressModeU: "repeat",
      addressModeV: "repeat",
      magFilter: "linear",
      minFilter: "linear",
    };

    const textureOptionsGUI = gui.addFolder("Texture Options");
    textureOptionsGUI.closed = false;
    this.mipmapsController = textureOptionsGUI
      .add(textureOptions, "mimaps")
      .name("Mipmaps");
    this.addressModeUController = textureOptionsGUI
      .add(textureOptions, "addressModeU")
      .options(["repeat", "clamp-to-edge"])
      .name("addressModeU");
    this.addressModeVController = textureOptionsGUI
      .add(textureOptions, "addressModeV")
      .options(["repeat", "clamp-to-edge"])
      .name("addressModeV");
    this.magFilterController = textureOptionsGUI
      .add(textureOptions, "magFilter")
      .options(["linear", "nearest"])
      .name("magFilter");
    this.minFilterController = textureOptionsGUI
      .add(textureOptions, "minFilter")
      .options(["linear", "nearest"])
      .name("minFilter");
  }

  private run() {
    requestAnimationFrame(this.drawFrame);
  }

  private drawFrame(now: number) {
    fpsAvg.value = 1000 / (now - this.current);
    this.fpsController.setValue(fpsAvg.value.toFixed(1));
    this.current = now;

    const startTime = performance.now();

    // model matrix
    const model = mat4.identity();

    // view matrix
    const yaw = utils.degToRad(this.yawController.getValue());
    let eye = vec3.create(
      this.xController.getValue(),
      this.yController.getValue(),
      this.zController.getValue()
    );
    const rotation = mat4.rotateY(mat4.identity(), yaw);
    const rotatedEye = vec3.transformMat4(eye, rotation);
    const target = vec3.create(0.0, 0.0, 0.0);
    let up = vec3.create(0.0, 1.0, 0.0);

    const view = mat4.lookAt(rotatedEye, target, up);

    // projection matrix
    const aspect = this.canvas.width / this.canvas.height;
    const projection = mat4.perspective(
      utils.degToRad(this.fovYController.getValue()),
      aspect,
      1.0,
      100.0
    );

    // model-view-projection matrix
    const mvp = mat4.multiply(projection, mat4.multiply(view, model));
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      mvp.buffer,
      mvp.byteOffset,
      mvp.byteLength
    );

    // texture index
    const textureIndex = this.mipmapsController.getValue() ? 1 : 0;

    // sampler index
    const addressModeU = this.addressModeUController.getValue();
    const addressModeV = this.addressModeVController.getValue();
    const magFilter = this.magFilterController.getValue();
    const minFilter = this.minFilterController.getValue();
    const samplerIndex =
      (addressModeU === "repeat" ? 1 : 0) +
      (addressModeV === "repeat" ? 2 : 0) +
      (magFilter === "linear" ? 4 : 0) +
      (minFilter === "linear" ? 8 : 0);

    const commandEncoder = this.device.createCommandEncoder({
      label: "GPU Command Encoder: Draw Frame",
    });

    const renderTargetTextureView = this.context
      .getCurrentTexture()
      .createView({ label: "GPU Texture View: Canvas Texture View" });
    const colorAttachments: GPURenderPassColorAttachment[] = [
      {
        view: renderTargetTextureView,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ];

    const renderPassDescriptor: GPURenderPassDescriptor = {
      label: "GPU Renderpass Descriptor: Draw Frame",
      colorAttachments,
      depthStencilAttachment: {
        view: this.depthTextureView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    };

    if (this.hasTimestamp) {
      renderPassDescriptor.timestampWrites = {
        querySet: this.querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      };
    }

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.renderPipeline);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, "uint32");
    passEncoder.setBindGroup(0, this.bindGroups[0][0]);
    passEncoder.setBindGroup(1, this.bindGroups[1][textureIndex]);
    passEncoder.setBindGroup(2, this.bindGroups[2][samplerIndex]);
    passEncoder.drawIndexed(36);

    passEncoder.end();

    if (this.hasTimestamp) {
      commandEncoder.resolveQuerySet(
        this.querySet,
        0,
        this.querySet.count,
        this.resolveBuffer,
        0
      );

      if ("unmapped" === this.resultBuffer.mapState) {
        commandEncoder.copyBufferToBuffer(
          this.resolveBuffer,
          0,
          this.resultBuffer,
          0,
          this.resultBuffer.size
        );
      }
    }

    this.device.queue.submit([commandEncoder.finish()]);

    cpuTimeAvg.value = performance.now() - startTime;
    this.cpuTimeController.setValue(cpuTimeAvg.value.toFixed(1));

    if (this.hasTimestamp && "unmapped" === this.resultBuffer.mapState) {
      this.resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const times = new BigInt64Array(this.resultBuffer.getMappedRange());
        gpuTimeAvg.value = Number(times[1] - times[0]) / 1000;
        this.gpuTimeController.setValue(gpuTimeAvg.value.toFixed(1));

        this.resultBuffer.unmap();
      });
    }

    requestAnimationFrame(this.drawFrame);
  }
}
