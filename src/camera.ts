import { mat4, Mat4, quat, Quat, utils, vec3, Vec3 } from "wgpu-matrix";

enum MoveMode {
  NONE = "None",
  TUMBLE = "Tumble",
  TRACK = "Track",
  DOLLY = "Dolly",
}

export default class Camera {
  private controlled: boolean;
  private moveMode: MoveMode;
  private x: number;
  private y: number;
  private rotateAmplitude: number;
  private pitch: number;
  private yaw: number;
  private scrollAmplitude: number;
  private target: Vec3;
  private distance: number;
  private panAmplitude: number;
  private rotation: Quat;

  public constructor() {
    this.controlled = false;
    this.moveMode = MoveMode.NONE;
    this.rotateAmplitude = 0.1;
    this.x = 0;
    this.y = 0;
    this.pitch = 0.0;
    this.yaw = 0.0;
    this.scrollAmplitude = 0.01;
    this.target = vec3.create(0.0, 0.0, 0.0);
    this.distance = 10.0;
    this.panAmplitude = 0.01;
    this.rotation = quat.create();

    this.onKeyDown = this.onKeyDown.bind(this);
    this.onKeyUp = this.onKeyUp.bind(this);
    this.onMouseDown = this.onMouseDown.bind(this);
    this.onMouseUp = this.onMouseUp.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);

    document.addEventListener("contextmenu", (event: MouseEvent) => {
      if (this.controlled) {
        event.preventDefault();
      }
    });
    document.addEventListener("keydown", this.onKeyDown);
    document.addEventListener("keyup", this.onKeyUp);
    document.addEventListener("mousedown", this.onMouseDown);
    document.addEventListener("mouseup", this.onMouseUp);
    document.addEventListener("mousemove", this.onMouseMove);
  }

  public get view(): Mat4 {
    const direction = vec3.normalize(
      vec3.transformQuat(vec3.create(0.0, 0.0, 1.0), this.rotation)
    );
    const eye = vec3.add(this.target, vec3.mulScalar(direction, this.distance));

    return mat4.lookAt(eye, this.target, vec3.create(0.0, 1.0, 0.0));
  }

  private onKeyDown(event: KeyboardEvent): void {
    switch (event.key) {
      case "Alt":
        this.controlled = true;
        break;
      default:
        break;
    }
  }

  private onKeyUp(event: KeyboardEvent): void {
    switch (event.key) {
      case "Alt":
        this.controlled = false;
        break;
      default:
        break;
    }
  }

  private onMouseDown(event: MouseEvent): void {
    if (!this.controlled) {
      return;
    }

    this.x = event.clientX;
    this.y = event.clientY;

    switch (event.button) {
      case 0:
        this.moveMode = MoveMode.TUMBLE;
        break;
      case 1:
        this.moveMode = MoveMode.TRACK;
        break;
      case 2:
        this.moveMode = MoveMode.DOLLY;
        break;
      default:
        this.moveMode = MoveMode.NONE;
        break;
    }
  }

  private onMouseUp(): void {
    if (!this.controlled) {
      return;
    }

    this.moveMode = MoveMode.NONE;
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.controlled || MoveMode.NONE === this.moveMode) {
      return;
    }

    const deltaX = event.clientX - this.x;
    this.x = event.clientX;

    const deltaY = event.clientY - this.y;
    this.y = event.clientY;

    switch (this.moveMode) {
      case MoveMode.TUMBLE:
        this.updateCameraRotation(deltaX, deltaY);
        break;
      case MoveMode.DOLLY:
        this.updateCameraDistance(deltaX, deltaY);
        break;
      case MoveMode.TRACK:
        this.updateCameraTarget(deltaX, deltaY);
        break;
      default:
        break;
    }
  }

  private updateCameraRotation(deltaX: number, deltaY: number): void {
    this.yaw -= deltaX * this.rotateAmplitude;
    this.pitch -= deltaY * this.rotateAmplitude;

    const quatYaw = quat.fromAxisAngle(
      vec3.create(0.0, 1.0, 0.0),
      utils.degToRad(this.yaw)
    );
    const quatPitch = quat.fromAxisAngle(
      vec3.create(1.0, 0.0, 0.0),
      utils.degToRad(this.pitch)
    );

    this.rotation = quat.mul(quatYaw, quatPitch);
  }

  private updateCameraDistance(deltaX: number, deltaY: number): void {
    this.distance -= (deltaX + deltaY) * this.scrollAmplitude;
  }

  private updateCameraTarget(deltaX: number, deltaY: number) {
    const panOffset = vec3.transformQuat(
      vec3.create(-deltaX * this.panAmplitude, deltaY * this.panAmplitude, 0.0),
      this.rotation
    );

    this.target = vec3.add(this.target, panOffset);
  }
}
