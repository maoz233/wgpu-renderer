import { Quat, vec3, Vec3 } from "wgpu-matrix";

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
  private theta: number;
  private phi: number;
  private target: Vec3;
  private distance: number;
  private rotation: Quat;

  public constructor() {
    this.controlled = false;
    this.moveMode = MoveMode.NONE;
    this.x = 0;
    this.y = 0;
    this.target = vec3.create(0.0, 0.0, 0.0);
    this.distance = 10.0;

    this.onKeyDown = this.onKeyDown.bind(this);
    this.onKeyUp = this.onKeyUp.bind(this);
    this.onMouseDown = this.onMouseDown.bind(this);
    this.onMouseUp = this.onMouseUp.bind(this);

    document.addEventListener("keydown", this.onKeyDown);
    document.addEventListener("keyup", this.onKeyUp);
    document.addEventListener("mousedown", this.onMouseDown);
    document.addEventListener("mouseup", this.onMouseUp);
    document.addEventListener("mousemove", this.onMouseMove);
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
        break;
    }

    if (MoveMode.NONE === this.moveMode) {
      return;
    }

    this.x = event.clientX;
    this.y = event.clientY;
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

    const deltaY = event.clientX - this.y;
    this.y = event.clientY;
  }
}
