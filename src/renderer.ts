import shaderCode from "@/shaders/shader.wgsl";
import mipmapShaderCode from "@/shaders/mipmap.wgsl";
import { mat4, utils } from "wgpu-matrix";
import { GUI, GUIController } from "dat.gui";
import { loadImageBitmap, rand, calculateMipLevelCount } from "@/utils";

type Transform = {
  offset: number[];
  scale: number;
};

export default class Renderer {
  private adapter: GPUAdapter;
  private device: GPUDevice;
  private canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private presentationFormat: GPUTextureFormat;
  private renderPipeline: GPURenderPipeline;
  private vertexBuffer: GPUBuffer;
  private indexBuffer: GPUBuffer;
  private bindGroups: Array<Array<GPUBindGroup>>;
  private transforms: Array<Transform>;
  private uniformBuffers: Array<GPUBuffer>;
  private current: number;
  private profilerController: GUIController;
  private mipmapsController: GUIController;
  private addressModeUController: GUIController;
  private addressModeVController: GUIController;
  private magFilterController: GUIController;
  private minFilterController: GUIController;

  public constructor() {
    this.transforms = new Array<Transform>();
    this.uniformBuffers = new Array<GPUBuffer>();
    this.bindGroups = new Array<Array<GPUBindGroup>>(
      new Array<GPUBindGroup>(),
      new Array<GPUBindGroup>(),
      new Array<GPUBindGroup>()
    );
    this.current = Date.now();
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
    this.device = await this.adapter?.requestDevice();
    if (!this.device) {
      throw Error("Failed to request WebGPU device.");
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

  private createRenderPipeline() {
    const shaderModule = this.createShaderModule("Shader Module", shaderCode);

    this.renderPipeline = this.device.createRenderPipeline({
      label: "Render Pipeline",
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
    const vertexCount = 4;
    const vertexComponents = 4;
    const texCoordComponents = 2;
    const unitSize =
      (vertexComponents + texCoordComponents) * Float32Array.BYTES_PER_ELEMENT;

    const data = [
      {
        vertex: [-0.5, 0.5, 0.0, 1.0],
        texCoord: [0, 0],
      },
      {
        vertex: [-0.5, -0.5, 0.0, 1.0],
        texCoord: [0, 1],
      },
      {
        vertex: [0.5, 0.5, 0.0, 1.0],
        texCoord: [1, 0],
      },
      {
        vertex: [0.5, -0.5, 0.0, 1.0],
        texCoord: [1, 1],
      },
    ];

    let vertexOffset = 0;
    let texCoordOffset = vertexComponents;
    const vertices = new Float32Array(vertexCount * unitSize);
    for (let i = 0; i < vertexCount; ++i) {
      vertices.set(data[i].vertex, vertexOffset);
      vertices.set(data[i].texCoord, texCoordOffset);
      vertexOffset += vertexComponents + texCoordComponents;
      texCoordOffset += vertexComponents + texCoordComponents;
    }

    this.vertexBuffer = this.createBuffer(
      "Vertex Buffer",
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
    const indices = new Uint32Array([0, 1, 2, 2, 1, 3]);

    this.indexBuffer = this.createBuffer(
      "Index Buffer",
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
    for (let i = 0; i < 10; ++i) {
      this.transforms.push({
        offset: [rand(-0.9, 0.9), rand(-0.9, 0.9)],
        scale: rand(0.2, 0.5),
      });

      const uniformBuffer = this.createBuffer(
        `Uniform Buffer ${i}`,
        4 * 4 * Float32Array.BYTES_PER_ELEMENT,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      );
      this.uniformBuffers.push(uniformBuffer);

      const bindGroup = this.device.createBindGroup({
        label: `Bind Group 0: ${i}`,
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: uniformBuffer,
            },
          },
        ],
      });
      this.bindGroups[0].push(bindGroup);
    }
  }

  private async createTexture() {
    const imageBitMap = await loadImageBitmap("images/f-texture.png");

    for (let i = 0; i < 2; ++i) {
      const mipLevelCount = i
        ? calculateMipLevelCount(imageBitMap.width, imageBitMap.height)
        : 1;

      const texture = this.createTextureFromSource(
        `2D Texture ${i}`,
        imageBitMap,
        mipLevelCount
      );

      const textureView = texture.createView({
        label: `Texture View ${i}`,
      });

      const bindGroup = this.device.createBindGroup({
        label: `Bind Group 1: ${i}`,
        layout: this.renderPipeline.getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: textureView,
          },
        ],
      });
      this.bindGroups[1].push(bindGroup);
    }

    for (let i = 0; i < 16; ++i) {
      const sampler = this.device.createSampler({
        label: `Texture Sampler`,
        addressModeU: i & 1 ? "repeat" : "clamp-to-edge",
        addressModeV: i & 2 ? "repeat" : "clamp-to-edge",
        magFilter: i & 4 ? "linear" : "nearest",
        minFilter: i & 8 ? "linear" : "nearest",
      });

      const bindGroup = this.device.createBindGroup({
        label: `Bind Group 2: ${i}`,
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
    source: ImageBitmap,
    mipLevelCount: number
  ) {
    const texture = this.device.createTexture({
      label,
      format: "rgba8unorm",
      mipLevelCount,
      size: [source.width, source.height],
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
    source: ImageBitmap,
    texture: GPUTexture
  ) {
    device.queue.copyExternalImageToTexture(
      { source },
      { texture },
      { width: source.width, height: source.height }
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
          label: "Mipmap Shader Module",
          code: mipmapShaderCode,
        });
        sampler = device.createSampler({
          minFilter: "linear",
        });
      }

      if (!pipelineByFormat.has(texture.format)) {
        const pipeline = device.createRenderPipeline({
          label: "Mipmap Generation Pipeline",
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
        "Mipmaps Generation Vertex Buffer",
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
        "Mipmap Generation Index Buffer",
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
        label: "Mipmap Generation Command Encoder",
      });

      let width = texture.width;
      let height = texture.height;
      let baseMipLevel = 0;
      while (width > 1 || height > 1) {
        width = Math.max(1, width / 2);
        height = Math.max(1, height / 2);

        const bindGroup = device.createBindGroup({
          label: "Mipmap Generation Bind Group",
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
          label: "Mipmap Generation Render Pass Descriptor",
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
    const profiler = { fps: "0" };
    const settings = {
      mimaps: false,
      addressModeU: "repeat",
      addressModeV: "repeat",
      magFilter: "linear",
      minFilter: "linear",
    };
    const gui = new GUI({
      name: "My GUI",
      autoPlace: true,
      hideable: true,
    });
    this.profilerController = gui.add(profiler, "fps").name("FPS");
    const settingsGUI = gui.addFolder("Settings");
    settingsGUI.closed = false;
    this.mipmapsController = settingsGUI
      .add(settings, "mimaps")
      .name("Mipmaps");
    this.addressModeUController = settingsGUI
      .add(settings, "addressModeU")
      .options(["repeat", "clamp-to-edge"])
      .name("addressModeU");
    this.addressModeVController = settingsGUI
      .add(settings, "addressModeV")
      .options(["repeat", "clamp-to-edge"])
      .name("addressModeV");
    this.magFilterController = settingsGUI
      .add(settings, "magFilter")
      .options(["linear", "nearest"])
      .name("magFilter");
    this.minFilterController = settingsGUI
      .add(settings, "minFilter")
      .options(["linear", "nearest"])
      .name("minFilter");
  }

  private run() {
    requestAnimationFrame(this.drawFrame);
  }

  private drawFrame() {
    const now = Date.now();
    this.profilerController.setValue((1000 / (now - this.current)).toFixed(2));
    this.current = now;

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

    // model-view-projectin matrix
    const aspect = this.canvas.width / this.canvas.height;
    const view = mat4.lookAt(
      [0.0, 0.0, -3.0],
      [0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0]
    );
    const projection = mat4.perspective(utils.degToRad(30), aspect, 1.0, 100.0);

    const commandEncoder = this.device.createCommandEncoder({
      label: "Draw Frame Command Encoder",
    });
    const renderTargetTextureView = this.context
      .getCurrentTexture()
      .createView({ label: "Render Target Texture View" });
    const colorAttachments: GPURenderPassColorAttachment[] = [
      {
        view: renderTargetTextureView,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ];

    const renderPassDescriptor: GPURenderPassDescriptor = {
      label: "Renderpass Descriptor",
      colorAttachments,
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.renderPipeline);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, "uint32");

    this.transforms.forEach(
      ({ offset, scale }: Transform, transformIndex: number) => {
        const scaleMat = mat4.scale(mat4.identity(), [
          scale / aspect,
          scale,
          1.0,
          1.0,
        ]);
        const translateMat = mat4.translate(mat4.identity(), [
          ...offset,
          1.0,
          1.0,
        ]);
        const rotateMat = mat4.rotateY(
          mat4.identity(),
          (this.current / 1000) % 360
        );

        const model = mat4.mul(scaleMat, mat4.mul(translateMat, rotateMat));
        const transform = mat4.multiply(projection, mat4.multiply(view, model));
        this.device.queue.writeBuffer(
          this.uniformBuffers[transformIndex],
          0,
          transform.buffer,
          transform.byteOffset,
          transform.byteLength
        );

        passEncoder.setBindGroup(0, this.bindGroups[0][transformIndex]);
        passEncoder.setBindGroup(1, this.bindGroups[1][textureIndex]);
        passEncoder.setBindGroup(2, this.bindGroups[2][samplerIndex]);
        passEncoder.drawIndexed(6);
      }
    );

    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(this.drawFrame);
  }
}
