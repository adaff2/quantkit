import numpy as np
import sdl2
import sdl2.ext
import sdl2.sdlttf as ttf
import ctypes


class Renderer:
    """Base SDL2 renderer with event loop."""

    def __init__(self, title: str = "Stochastic Visualizer", width: int = 1280, height: int = 720):
        self.title = title.encode()
        self.width = width
        self.height = height
        self._window = None
        self._renderer = None

    def __enter__(self):
        sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)
        self._window = sdl2.SDL_CreateWindow(
            self.title,
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            self.width,
            self.height,
            sdl2.SDL_WINDOW_SHOWN
        )
        self._renderer = sdl2.SDL_CreateRenderer(
            self._window, -1,
            sdl2.SDL_RENDERER_ACCELERATED | sdl2.SDL_RENDERER_PRESENTVSYNC
        )
        return self

    def __exit__(self, *args):
        sdl2.SDL_DestroyRenderer(self._renderer)
        sdl2.SDL_DestroyWindow(self._window)
        sdl2.SDL_Quit()

    def clear(self, r=15, g=15, b=15):
        """Clear screen with background color."""
        sdl2.SDL_SetRenderDrawColor(self._renderer, r, g, b, 255)
        sdl2.SDL_RenderClear(self._renderer)

    def present(self):
        sdl2.SDL_RenderPresent(self._renderer)

    def poll_events(self) -> bool:
        """Returns False if user closes the window."""
        event = sdl2.SDL_Event()
        while sdl2.SDL_PollEvent(ctypes.byref(event)):
            if event.type == sdl2.SDL_QUIT:
                return False
            if event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    return False
        return True


class PathPlot:
    """
    Plots Monte Carlo paths using SDL2.
    Much faster than matplotlib for large number of paths.
    """

    def __init__(
        self,
        renderer: Renderer,
        margin: int = 60,
        font_path: str = "C:/Windows/Fonts/consola.ttf",
        font_size: int = 12,
    ):
        self.renderer = renderer
        self.margin = margin
        self.font_path = font_path.encode()
        self.font_size = font_size
        self._font = None
        self._initial_view = None
        self._view = None

    def _init_font(self):
        ttf.TTF_Init()
        self._font = ttf.TTF_OpenFont(self.font_path, self.font_size)

    def _close_font(self):
        if self._font:
            ttf.TTF_CloseFont(self._font)
        ttf.TTF_Quit()

    def _draw_text(self, text: str, x: int, y: int, color: tuple = (200, 200, 200)):
        if not self._font:
            return
        r, g, b = color
        sdl_color = ttf.SDL_Color(r, g, b, 255)
        surface = ttf.TTF_RenderText_Blended(self._font, text.encode(), sdl_color)
        if not surface:
            return
        texture = sdl2.SDL_CreateTextureFromSurface(self.renderer._renderer, surface)
        sdl2.SDL_FreeSurface(surface)
        if not texture:
            return

        w, h = ctypes.c_int(0), ctypes.c_int(0)
        sdl2.SDL_QueryTexture(texture, None, None, ctypes.byref(w), ctypes.byref(h))
        dst = sdl2.SDL_Rect(x, y, w.value, h.value)
        sdl2.SDL_RenderCopy(self.renderer._renderer, texture, None, dst)
        sdl2.SDL_DestroyTexture(texture)

    def _clamp_view(self):
        """Clamp current view so user cannot zoom out past initial bounds."""
        if self._initial_view is None or self._view is None:
            return

        x0, x1 = self._initial_view["x_min"], self._initial_view["x_max"]
        y0, y1 = self._initial_view["y_min"], self._initial_view["y_max"]
        vx0, vx1 = self._view["x_min"], self._view["x_max"]
        vy0, vy1 = self._view["y_min"], self._view["y_max"]

        init_x_span = x1 - x0
        init_y_span = y1 - y0

        # Prevent extreme zoom-in and divisions by near-zero spans.
        min_x_span = max(init_x_span * 1e-4, 1e-9)
        min_y_span = max(init_y_span * 1e-4, 1e-9)

        x_span = max(vx1 - vx0, min_x_span)
        y_span = max(vy1 - vy0, min_y_span)

        # Do not allow zoom-out beyond initial range.
        x_span = min(x_span, init_x_span)
        y_span = min(y_span, init_y_span)

        # Clamp x window into initial range.
        if vx0 < x0:
            vx0 = x0
            vx1 = vx0 + x_span
        elif vx1 > x1:
            vx1 = x1
            vx0 = vx1 - x_span
        else:
            center_x = 0.5 * (vx0 + vx1)
            vx0 = center_x - 0.5 * x_span
            vx1 = center_x + 0.5 * x_span

        # Clamp y window into initial range.
        if vy0 < y0:
            vy0 = y0
            vy1 = vy0 + y_span
        elif vy1 > y1:
            vy1 = y1
            vy0 = vy1 - y_span
        else:
            center_y = 0.5 * (vy0 + vy1)
            vy0 = center_y - 0.5 * y_span
            vy1 = center_y + 0.5 * y_span

        self._view["x_min"], self._view["x_max"] = vx0, vx1
        self._view["y_min"], self._view["y_max"] = vy0, vy1

    def _zoom_at(self, mouse_x: int, mouse_y: int, zoom_factor: float):
        """Zoom around current mouse position in plot area."""
        if self._view is None:
            return

        left = self.margin
        top = self.margin
        plot_w = self.renderer.width - 2 * self.margin
        plot_h = self.renderer.height - 2 * self.margin

        rel_x = (mouse_x - left) / plot_w if plot_w > 0 else 0.5
        rel_y = (mouse_y - top) / plot_h if plot_h > 0 else 0.5
        rel_x = float(np.clip(rel_x, 0.0, 1.0))
        rel_y = float(np.clip(rel_y, 0.0, 1.0))

        vx0, vx1 = self._view["x_min"], self._view["x_max"]
        vy0, vy1 = self._view["y_min"], self._view["y_max"]
        x_span = vx1 - vx0
        y_span = vy1 - vy0

        x_data = vx0 + rel_x * x_span
        y_data = vy1 - rel_y * y_span

        new_x_span = x_span * zoom_factor
        new_y_span = y_span * zoom_factor

        new_x_min = x_data - rel_x * new_x_span
        new_x_max = new_x_min + new_x_span
        new_y_max = y_data + rel_y * new_y_span
        new_y_min = new_y_max - new_y_span

        self._view["x_min"], self._view["x_max"] = new_x_min, new_x_max
        self._view["y_min"], self._view["y_max"] = new_y_min, new_y_max
        self._clamp_view()

    def _to_screen(self, paths: np.ndarray, x_values: np.ndarray):
        """Normalize paths to screen coordinates."""
        w = self.renderer.width - 2 * self.margin
        h = self.renderer.height - 2 * self.margin

        x_min, x_max = self._view["x_min"], self._view["x_max"]
        y_min, y_max = self._view["y_min"], self._view["y_max"]
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0

        xs = (self.margin + w * (x_values - x_min) / x_range).astype(np.int32)
        ys = (self.margin + h * (1 - (paths - y_min) / y_range)).astype(np.int32)

        return xs, ys

    def _mouse_in_plot_area(self, mouse_x: int, mouse_y: int) -> bool:
        left = self.margin
        top = self.margin
        right = self.renderer.width - self.margin
        bottom = self.renderer.height - self.margin
        return left <= mouse_x <= right and top <= mouse_y <= bottom

    def _screen_to_world(self, mouse_x: int, mouse_y: int):
        """Convert screen coordinates to world/data coordinates based on current view."""
        left = self.margin
        top = self.margin
        plot_w = self.renderer.width - 2 * self.margin
        plot_h = self.renderer.height - 2 * self.margin

        if plot_w <= 0 or plot_h <= 0:
            return None, None

        rel_x = float(np.clip((mouse_x - left) / plot_w, 0.0, 1.0))
        rel_y = float(np.clip((mouse_y - top) / plot_h, 0.0, 1.0))

        x_min, x_max = self._view["x_min"], self._view["x_max"]
        y_min, y_max = self._view["y_min"], self._view["y_max"]

        world_x = x_min + rel_x * (x_max - x_min)
        world_y = y_max - rel_y * (y_max - y_min)
        return world_x, world_y

    def _draw_crosshair(self, mouse_x: int, mouse_y: int):
        """Draw mouse-centered crosshair and world coordinates."""
        if not self._mouse_in_plot_area(mouse_x, mouse_y):
            return

        world_x, world_y = self._screen_to_world(mouse_x, mouse_y)
        if world_x is None or world_y is None:
            return

        left = self.margin
        top = self.margin
        right = self.renderer.width - self.margin
        bottom = self.renderer.height - self.margin
        rdr = self.renderer._renderer

        # Crosshair in warm yellow so it stays visible over cyan paths.
        sdl2.SDL_SetRenderDrawColor(rdr, 255, 220, 120, 255)
        sdl2.SDL_RenderDrawLine(rdr, left, mouse_y, right, mouse_y)
        sdl2.SDL_RenderDrawLine(rdr, mouse_x, top, mouse_x, bottom)

        text = f"x={world_x:.4f}, y={world_y:.4f}"
        label_x = min(mouse_x + 12, right - 190)
        label_y = max(mouse_y - 20, top + 2)
        self._draw_text(text, label_x, label_y, color=(255, 220, 120))

    def _draw_axes(self, x_ticks: int, y_ticks: int):
        """Draw axes, tick marks, grid lines and labels."""
        w = self.renderer.width - 2 * self.margin
        h = self.renderer.height - 2 * self.margin
        rdr = self.renderer._renderer

        x_min, x_max = self._view["x_min"], self._view["x_max"]
        y_min, y_max = self._view["y_min"], self._view["y_max"]

        # axes color: light grey
        sdl2.SDL_SetRenderDrawColor(rdr, 200, 200, 200, 255)

        # x axis (bottom)
        sdl2.SDL_RenderDrawLine(rdr,
            self.margin, self.margin + h,
            self.margin + w, self.margin + h
        )
        # y axis (left)
        sdl2.SDL_RenderDrawLine(rdr,
            self.margin, self.margin,
            self.margin, self.margin + h
        )

        # x ticks + vertical grid lines + labels
        for i in range(x_ticks + 1):
            x = self.margin + int(i * w / x_ticks)
            x_val = x_min + i * (x_max - x_min) / x_ticks

            # grid line
            sdl2.SDL_SetRenderDrawColor(rdr, 60, 60, 60, 255)
            sdl2.SDL_RenderDrawLine(rdr, x, self.margin, x, self.margin + h)

            # tick mark
            sdl2.SDL_SetRenderDrawColor(rdr, 200, 200, 200, 255)
            sdl2.SDL_RenderDrawLine(rdr, x, self.margin + h - 4, x, self.margin + h + 4)

            # label
            label = f"{x_val:.2f}"
            self._draw_text(label, x - 15, self.margin + h + 8)

        # y ticks + horizontal grid lines + labels
        for i in range(y_ticks + 1):
            y = self.margin + int(i * h / y_ticks)
            y_val = y_max - i * (y_max - y_min) / y_ticks

            # grid line
            sdl2.SDL_SetRenderDrawColor(rdr, 60, 60, 60, 255)
            sdl2.SDL_RenderDrawLine(rdr, self.margin, y, self.margin + w, y)

            # tick mark
            sdl2.SDL_SetRenderDrawColor(rdr, 200, 200, 200, 255)
            sdl2.SDL_RenderDrawLine(rdr, self.margin - 4, y, self.margin + 4, y)

            # label
            label = f"{y_val:.2f}"
            self._draw_text(label, 2, y - 6)

    def draw(
        self,
        paths: np.ndarray,
        x_values: np.ndarray,
        color: tuple = (0, 180, 255),
        alpha: int = 30,
        x_ticks: int = 10,
        y_ticks: int = 8,
        show_crosshair: bool = False,
        mouse_x: int = 0,
        mouse_y: int = 0,
        cycle_colors: bool = False,
    ):
        """
        Draw all paths at once with axes and labels.
        :param paths: (n_paths x n_points) matrix
        :param x_values: x coordinates corresponding to path points
        :param color: RGB tuple
        :param alpha: transparency (0-255), lower = more transparent
        :param x_ticks: number of ticks on x axis
        :param y_ticks: number of ticks on y axis
        :param show_crosshair: whether to draw crosshair/coordinates
        :param mouse_x: current mouse x in screen space
        :param mouse_y: current mouse y in screen space
        """
        self._draw_axes(x_ticks, y_ticks)

        xs, ys = self._to_screen(paths, x_values)
        r, g, b = color

        colors = [(255, 0, 0), (0, 255, 0), (0, 180, 255), (255, 120, 0), (255, 0, 255)]

        if cycle_colors:
            # Vary color across paths for better visibility when many paths overlap.
            n_paths = paths.shape[0]
            for p in range(n_paths):
                path_alpha = int(alpha * (0.5 + 0.5 * p / max(n_paths - 1, 1)))
                random_color = colors[p % len(colors)]
                sdl2.SDL_SetRenderDrawColor(self.renderer._renderer, *random_color, path_alpha)
                for i in range(paths.shape[1] - 1):
                    sdl2.SDL_RenderDrawLine(
                        self.renderer._renderer,
                        int(xs[i]),   int(ys[p, i]),
                        int(xs[i+1]), int(ys[p, i+1])
                    )
        else:
            sdl2.SDL_SetRenderDrawColor(self.renderer._renderer, r, g, b, alpha)

            n_paths, n_points = paths.shape
            for p in range(n_paths):
                for i in range(n_points - 1):
                    sdl2.SDL_RenderDrawLine(
                        self.renderer._renderer,
                        int(xs[i]),   int(ys[p, i]),
                        int(xs[i+1]), int(ys[p, i+1])
                    )

        if show_crosshair:
            self._draw_crosshair(mouse_x, mouse_y)

    def show(
        self,
        paths: np.ndarray,
        x_start: int = 0,
        x_end: int = 1,
        color: tuple = (0, 180, 255),
        alpha: int = 30,
        x_ticks: int = 10,
        y_ticks: int = 8,
        cycle_colors: bool = False,
    ):
        """Open window and display paths until user closes it."""
        with self.renderer as r:
            self._init_font()
            y_min, y_max = float(paths.min()), float(paths.max())
            if np.isclose(y_min, y_max):
                y_min -= 1.0
                y_max += 1.0

            x_values = np.linspace(x_start, x_end, paths.shape[1], dtype=np.float64)
            self._initial_view = {
                "x_min": float(x_start),
                "x_max": float(x_end),
                "y_min": y_min,
                "y_max": y_max,
            }
            self._view = dict(self._initial_view)

            running = True
            left_shift_pressed = False
            while running:
                event = sdl2.SDL_Event()
                while sdl2.SDL_PollEvent(ctypes.byref(event)):
                    if event.type == sdl2.SDL_QUIT:
                        running = False
                    elif event.type == sdl2.SDL_KEYDOWN:
                        if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                            running = False
                        elif event.key.keysym.sym == sdl2.SDLK_r:
                            self._view = dict(self._initial_view)
                        elif event.key.keysym.sym == sdl2.SDLK_LSHIFT:
                            left_shift_pressed = True
                    elif event.type == sdl2.SDL_KEYUP:
                        if event.key.keysym.sym == sdl2.SDLK_LSHIFT:
                            left_shift_pressed = False
                    elif event.type == sdl2.SDL_MOUSEWHEEL:
                        mouse_x = ctypes.c_int(0)
                        mouse_y = ctypes.c_int(0)
                        sdl2.SDL_GetMouseState(ctypes.byref(mouse_x), ctypes.byref(mouse_y))
                        if event.wheel.y > 0:
                            self._zoom_at(mouse_x.value, mouse_y.value, 0.9)
                        elif event.wheel.y < 0:
                            self._zoom_at(mouse_x.value, mouse_y.value, 1.1)

                mouse_x = ctypes.c_int(0)
                mouse_y = ctypes.c_int(0)
                sdl2.SDL_GetMouseState(ctypes.byref(mouse_x), ctypes.byref(mouse_y))

                r.clear()
                self.draw(
                    paths,
                    x_values,
                    color,
                    alpha,
                    x_ticks,
                    y_ticks,
                    cycle_colors=cycle_colors,
                    show_crosshair=left_shift_pressed,
                    mouse_x=mouse_x.value,
                    mouse_y=mouse_y.value,
                )
                r.present()
            self._close_font()
