import shaderCode from "@/shaders/shader.wgsl";
import { mat4 } from "wgpu-matrix";
import { GUI, GUIController } from "dat.gui";
import { rand } from "@/utils";

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
  private texture: GPUTexture;
  private textureView: GPUTextureView;
  private sampler: GPUSampler;
  private transforms: Array<Transform>;
  private uniformBuffers: Array<GPUBuffer>;
  private bindGroups: Array<GPUBindGroup>;
  private current: number;
  private profilerController: GUIController;

  public constructor() {
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
    this.createTexture();
    this.createUniformBuffer();
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

  private createTexture() {
    const width = 5;
    const height = 7;
    const colorComponents = 4;
    const r = [255, 0, 0, 255];
    const g = [0, 255, 0, 255];
    const b = [0, 0, 255, 255];

    // prettier-ignore
    const textureData = new Uint8Array([
      b, r, r, r, r,
      r, g, g, g, r,
      r, g, r, r, r,
      r, g, g, r, r,
      r, g, r, r, r,
      r, g, r, r, r,
      r, r, r, r, r,
    ].flat());

    this.texture = this.device.createTexture({
      label: "2D Texture",
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      {
        texture: this.texture,
      },
      textureData,
      {
        bytesPerRow: width * colorComponents * Uint8Array.BYTES_PER_ELEMENT,
      },
      {
        width,
        height,
      }
    );

    this.textureView = this.texture.createView({
      label: "Texture View",
    });

    this.sampler = this.device.createSampler({
      label: `Texture Sampler`,
    });
  }

  private createUniformBuffer() {
    this.transforms = new Array<Transform>();
    this.uniformBuffers = new Array<GPUBuffer>();
    this.bindGroups = new Array<GPUBindGroup>();

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
        label: `Bind Group ${i}`,
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: uniformBuffer,
            },
          },
          {
            binding: 1,
            resource: this.sampler,
          },
          {
            binding: 2,
            resource: this.textureView,
          },
        ],
      });
      this.bindGroups.push(bindGroup);
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

  private initGUI() {
    const profiler = { fps: "0" };
    const gui = new GUI({
      name: "My GUI",
      autoPlace: true,
      hideable: true,
    });
    this.profilerController = gui.add(profiler, "fps").name("FPS");
  }

  private run() {
    requestAnimationFrame(this.drawFrame);
  }

  private drawFrame() {
    const now = Date.now();
    this.profilerController.setValue((1000 / (now - this.current)).toFixed(2));
    this.current = now;

    // model-view-projectin matrix
    const aspect = this.canvas.width / this.canvas.height;
    const view = mat4.lookAt(
      [0.0, 0.0, -3.0],
      [0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0]
    );
    const projection = mat4.perspective((2 * Math.PI) / 12, aspect, 1.0, 100.0);

    const commandEncoder = this.device.createCommandEncoder();
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

    this.transforms.forEach(({ offset, scale }: Transform, index: number) => {
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
        this.uniformBuffers[index],
        0,
        transform.buffer,
        transform.byteOffset,
        transform.byteLength
      );

      passEncoder.setBindGroup(0, this.bindGroups[index]);
      passEncoder.drawIndexed(6);
    });

    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(this.drawFrame);
  }
}
