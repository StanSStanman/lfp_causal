import numpy as np
import matplotlib.pyplot as pl


class DraggableColorbar(object):
    """Enable interactive colorbar.
    See http://www.ster.kuleuven.be/~pieterd/python/html/plotting/interactive_colorbar.html
    """  # noqa: E501

    def __init__(self, cbar, mappable):
        import matplotlib.pyplot as plt
        self.cbar = cbar
        self.mappable = mappable
        self.press = None
        self.cycle = sorted([i for i in dir(plt.cm) if
                             hasattr(getattr(plt.cm, i), 'N')])
        self.cycle += [mappable.get_cmap().name]
        self.index = self.cycle.index(mappable.get_cmap().name)
        self.lims = (self.cbar.norm.vmin, self.cbar.norm.vmax)
        self.connect()

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.cbar.patch.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.cbar.patch.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.cbar.patch.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.keypress = self.cbar.patch.figure.canvas.mpl_connect(
            'key_press_event', self.key_press)
        self.scroll = self.cbar.patch.figure.canvas.mpl_connect(
            'scroll_event', self.on_scroll)

    def on_press(self, event):
        """Handle button press."""
        if event.inaxes != self.cbar.ax:
            return
        self.press = event.y

    def key_press(self, event):
        """Handle key press."""
        # print(event.key)
        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.key == 'down':
            self.index += 1
        elif event.key == 'up':
            self.index -= 1
        elif event.key == ' ':  # space key resets scale
            self.cbar.norm.vmin = self.lims[0]
            self.cbar.norm.vmax = self.lims[1]
        elif event.key == '+':
            self.cbar.norm.vmin -= (perc * scale) * -1
            self.cbar.norm.vmax += (perc * scale) * -1
        elif event.key == '-':
            self.cbar.norm.vmin -= (perc * scale) * 1
            self.cbar.norm.vmax += (perc * scale) * 1
        elif event.key == 'pageup':
            self.cbar.norm.vmin -= (perc * scale) * 1
            self.cbar.norm.vmax -= (perc * scale) * 1
        elif event.key == 'pagedown':
            self.cbar.norm.vmin -= (perc * scale) * -1
            self.cbar.norm.vmax -= (perc * scale) * -1
        else:
            return
        if self.index < 0:
            self.index = len(self.cycle) - 1
        elif self.index >= len(self.cycle):
            self.index = 0
        cmap = self.cycle[self.index]
        self.cbar.mappable.set_cmap(cmap)
        self.cbar.draw_all()
        self.mappable.set_cmap(cmap)
        self._update()

    def on_motion(self, event):
        """Handle mouse movements."""
        if self.press is None:
            return
        if event.inaxes != self.cbar.ax:
            return
        yprev = self.press
        dy = event.y - yprev
        self.press = event.y
        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.button == 1:
            self.cbar.norm.vmin -= (perc * scale) * np.sign(dy)
            self.cbar.norm.vmax -= (perc * scale) * np.sign(dy)
        elif event.button == 3:
            self.cbar.norm.vmin -= (perc * scale) * np.sign(dy)
            self.cbar.norm.vmax += (perc * scale) * np.sign(dy)
        self._update()

    def on_release(self, event):
        """Handle release."""
        self.press = None
        self._update()

    def on_scroll(self, event):
        """Handle scroll."""
        scale = 1.1 if event.step < 0 else 1. / 1.1
        self.cbar.norm.vmin *= scale
        self.cbar.norm.vmax *= scale
        self._update()

    def _update(self):
        self.cbar.set_ticks(None, update_ticks=True)  # use default
        self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()
