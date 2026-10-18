// Shows wp-engine's wallpaper windows as the GNOME desktop background.
//
// GNOME has no wlr-layer-shell, so on GNOME wp-engine opens one plain
// xdg_toplevel per monitor (app id APP_ID, title TITLE_PREFIX + connector).
// This extension makes each such window invisible and paints a Clutter.Clone
// of it into the background group on its monitor. The window stays mapped at
// opacity 0 rather than hidden, so Mutter keeps sending it frame callbacks;
// a clone paints its source at the clone's own opacity.

import Clutter from 'gi://Clutter';
import Meta from 'gi://Meta';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import {Workspace} from 'resource:///org/gnome/shell/ui/workspace.js';
import {WorkspaceThumbnail} from 'resource:///org/gnome/shell/ui/workspaceThumbnail.js';
import {Extension, InjectionManager} from 'resource:///org/gnome/shell/extensions/extension.js';

// Keep in sync with XDG_APP_ID / XDG_TITLE_PREFIX in src/platform/wayland/mod.rs.
const APP_ID = 'io.github.code_wolf_byte.wp_engine.wallpaper';
const TITLE_PREFIX = 'wp-engine wallpaper @';

const isWallpaper = win => win?.get_wm_class?.() === APP_ID;

export default class WpEngineWallpaper extends Extension {
    enable() {
        // window -> {clone, signals: [[object, id]]}
        this._wallpapers = new Map();
        this._injections = new InjectionManager();

        // Out of alt-tab (altTab.js lists windows through get_tab_list) and
        // the overview / workspace thumbnails.
        this._injections.overrideMethod(Meta.Display.prototype, 'get_tab_list',
            orig => function (...args) {
                return orig.apply(this, args).filter(w => !isWallpaper(w));
            });
        for (const proto of [Workspace.prototype, WorkspaceThumbnail.prototype]) {
            this._injections.overrideMethod(proto, '_isOverviewWindow',
                orig => function (win) {
                    const metaWindow = win?.meta_window ?? win;
                    return !isWallpaper(metaWindow) && orig.call(this, win);
                });
        }

        // Runs after the shell's own map handler, so _track's _hide cancels
        // the map animation that would fade the window back in.
        this._mapSignal = global.window_manager.connect('map',
            (_wm, actor) => this._track(actor.meta_window));
        this._monitorsSignal = Main.layoutManager.connect('monitors-changed',
            () => this._wallpapers.forEach((_, win) => this._place(win)));

        for (const actor of global.get_window_actors())
            this._track(actor.meta_window);
    }

    disable() {
        global.window_manager.disconnect(this._mapSignal);
        Main.layoutManager.disconnect(this._monitorsSignal);
        for (const win of [...this._wallpapers.keys()])
            this._untrack(win, true);
        this._injections.clear();
        this._wallpapers = null;
        this._injections = null;
    }

    _track(win) {
        if (!isWallpaper(win) || this._wallpapers.has(win))
            return;
        const actor = win.get_compositor_private();
        if (!actor)
            return;
        const clone = new Clutter.Clone({source: actor});
        Main.layoutManager._backgroundGroup.add_child(clone);
        const entry = {clone, signals: []};
        this._wallpapers.set(win, entry);
        entry.signals.push([win, win.connect('notify::title', () => this._place(win))]);
        entry.signals.push([win, win.connect('unmanaged', () => this._untrack(win, false))]);

        win.stick();
        win.make_below();
        this._hide(actor);
        this._place(win);
    }

    _untrack(win, restore) {
        const entry = this._wallpapers.get(win);
        if (!entry)
            return;
        entry.signals.forEach(([obj, id]) => obj.disconnect(id));
        entry.clone.destroy();
        this._wallpapers.delete(win);
        const actor = win.get_compositor_private();
        if (restore && actor)
            actor.opacity = 255;
    }

    _hide(actor) {
        actor.remove_all_transitions();
        actor.opacity = 0;
    }

    /** Size the clone to the monitor named in the window title. */
    _place(win) {
        const entry = this._wallpapers.get(win);
        const title = win.get_title() ?? '';
        if (!entry || !title.startsWith(TITLE_PREFIX))
            return;
        const connector = title.slice(TITLE_PREFIX.length);
        let index = connector
            ? global.backend.get_monitor_manager().get_monitor_for_connector(connector)
            : -1;
        if (index < 0)
            index = Main.layoutManager.primaryIndex;
        const geometry = Main.layoutManager.monitors[index];
        if (!geometry)
            return;
        entry.clone.set_position(geometry.x, geometry.y);
        entry.clone.set_size(geometry.width, geometry.height);
        // Keep the real (invisible) window on its monitor too, so clicks on
        // the bare desktop there land on it.
        win.move_frame(false, geometry.x, geometry.y);
    }
}
