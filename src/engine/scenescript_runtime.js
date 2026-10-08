// SceneScript runtime: the globals Wallpaper Engine exposes to scene scripts,
// evaluated once into a persistent boa realm (see `engine::script`).
//
// The Rust side drives it through a handful of `__wp_*` entry points:
//   __wp_register_object(index, init)   one persistent layer proxy per object
//   __wp_register_instance(index, prop, kind, authoredProps, factory)
//   __wp_frame(runtime, timeOfDay, frametime)   timers + init/update, once per frame
//   __wp_collect()                      JSON diff of everything scripts changed
// Layer proxies are the source of truth for scripted state: Rust seeds them
// once, scripts read and write them, and `__wp_collect` reports what changed.

var __wp = {
    objects: [],
    instances: [],
    timers: [],
    nextTimer: 1,
    commands: [],
    logs: [],
    storage: {},
    storageDirty: false,
    audio: null,
    audioWanted: false,
    frame: 0
};

function __wp_num(v, d) { return (typeof v === 'number' && isFinite(v)) ? v : d; }

// ── Vectors ────────────────────────────────────────────────────────────────

function __wp_parse(s) {
    return String(s).trim().split(/\s+/).map(function (p) { return parseFloat(p) || 0; });
}

function Vec2(x, y) {
    if (typeof x === 'string') { var p = __wp_parse(x); x = p[0]; y = p[1]; }
    else if (x !== null && typeof x === 'object') { y = x.y; x = x.x; }
    this.x = __wp_num(x, 0); this.y = __wp_num(y, 0);
}
function Vec3(x, y, z) {
    if (typeof x === 'string') { var p = __wp_parse(x); x = p[0]; y = p[1]; z = p[2]; }
    else if (x !== null && typeof x === 'object') { z = x.z; y = x.y; x = x.x; }
    this.x = __wp_num(x, 0); this.y = __wp_num(y, 0); this.z = __wp_num(z, 0);
}
function Vec4(x, y, z, w) {
    if (typeof x === 'string') { var p = __wp_parse(x); x = p[0]; y = p[1]; z = p[2]; w = p[3]; }
    else if (x !== null && typeof x === 'object') { w = x.w; z = x.z; y = x.y; x = x.x; }
    this.x = __wp_num(x, 0); this.y = __wp_num(y, 0); this.z = __wp_num(z, 0); this.w = __wp_num(w, 0);
}

// Shared component-wise vector methods, generated per dimension.
(function () {
    var dims = [[Vec2, ['x', 'y']], [Vec3, ['x', 'y', 'z']], [Vec4, ['x', 'y', 'z', 'w']]];
    dims.forEach(function (d) {
        var C = d[0], k = d[1];
        function make(f) { var o = Object.create(C.prototype); k.forEach(function (c, i) { o[c] = f(c, i); }); return o; }
        function arg(o, c) { return (typeof o === 'number') ? o : __wp_num(o && o[c], 0); }
        var P = C.prototype;
        P.copy = function () { var s = this; return make(function (c) { return s[c]; }); };
        P.add = function (o) { var s = this; return make(function (c) { return s[c] + arg(o, c); }); };
        P.subtract = function (o) { var s = this; return make(function (c) { return s[c] - arg(o, c); }); };
        P.multiply = function (o) { var s = this; return make(function (c) { return s[c] * arg(o, c); }); };
        P.divide = function (o) { var s = this; return make(function (c) { return s[c] / arg(o, c); }); };
        P.negate = function () { var s = this; return make(function (c) { return -s[c]; }); };
        P.abs = function () { var s = this; return make(function (c) { return Math.abs(s[c]); }); };
        P.round = function () { var s = this; return make(function (c) { return Math.round(s[c]); }); };
        P.floor = function () { var s = this; return make(function (c) { return Math.floor(s[c]); }); };
        P.ceil = function () { var s = this; return make(function (c) { return Math.ceil(s[c]); }); };
        P.min = function (o) { var s = this; return make(function (c) { return Math.min(s[c], arg(o, c)); }); };
        P.max = function (o) { var s = this; return make(function (c) { return Math.max(s[c], arg(o, c)); }); };
        P.mix = function (o, t) { var s = this; return make(function (c) { return s[c] + (arg(o, c) - s[c]) * t; }); };
        P.dot = function (o) { var s = this, r = 0; k.forEach(function (c) { r += s[c] * arg(o, c); }); return r; };
        P.lengthSqr = function () { return this.dot(this); };
        P.length = function () { return Math.sqrt(this.lengthSqr()); };
        P.normalize = function () { var l = this.length(); return l > 0 ? this.divide(l) : this.copy(); };
        P.equals = function (o) { var s = this; return !!o && k.every(function (c) { return s[c] === o[c]; }); };
        P.reflect = function (n) { return this.subtract(n.multiply(2 * this.dot(n))); };
        P.toString = function () { var s = this; return k.map(function (c) { return s[c]; }).join(' '); };
    });
    Vec3.prototype.cross = function (o) {
        return new Vec3(this.y * o.z - this.z * o.y, this.z * o.x - this.x * o.z, this.x * o.y - this.y * o.x);
    };
    Vec2.prototype.perpendicular = function () { return new Vec2(-this.y, this.x); };
})();

// ── Math helpers ───────────────────────────────────────────────────────────

var WEMath = {
    PI: Math.PI,
    deg2rad: Math.PI / 180,
    rad2deg: 180 / Math.PI,
    smoothStep: function (a, b, x) {
        if (a === b) { return x < a ? 0 : 1; }
        var t = Math.min(Math.max((x - a) / (b - a), 0), 1);
        return t * t * (3 - 2 * t);
    },
    mix: function (a, b, t) { return a + (b - a) * t; },
    lerp: function (a, b, t) { return a + (b - a) * t; },
    clamp: function (x, a, b) { return Math.min(Math.max(x, a), b); },
    frac: function (x) { return x - Math.floor(x); },
    sign: function (x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }
};

var WEVector = {
    angleVector2: function (deg) { var r = deg * WEMath.deg2rad; return new Vec2(Math.cos(r), Math.sin(r)); },
    vectorAngle2: function (v) { return Math.atan2(v.y, v.x) * WEMath.rad2deg; }
};

var WEColor = {
    rgb2hsv: function (c) {
        var r = c.x, g = c.y, b = c.z, max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min, h = 0;
        if (d > 0) {
            if (max === r) { h = ((g - b) / d) % 6; }
            else if (max === g) { h = (b - r) / d + 2; }
            else { h = (r - g) / d + 4; }
            h /= 6; if (h < 0) { h += 1; }
        }
        return new Vec3(h, max === 0 ? 0 : d / max, max);
    },
    hsv2rgb: function (c) {
        var h = WEMath.frac(c.x) * 6, s = c.y, v = c.z, i = Math.floor(h), f = h - i;
        var p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
        switch (i % 6) {
            case 0: return new Vec3(v, t, p);
            case 1: return new Vec3(q, v, p);
            case 2: return new Vec3(p, v, t);
            case 3: return new Vec3(p, q, v);
            case 4: return new Vec3(t, p, v);
            default: return new Vec3(v, p, q);
        }
    },
    normalizeColor: function (c) { return new Vec3(c).divide(255); },
    expandColor: function (c) { return new Vec3(c).multiply(255); }
};

// ── Console, shared, media ─────────────────────────────────────────────────

function __wp_log(level, args) {
    if (__wp.logs.length < 64) {
        __wp.logs.push(level + ': ' + Array.prototype.map.call(args, function (a) {
            try { return typeof a === 'object' ? JSON.stringify(a) : String(a); } catch (e) { return String(a); }
        }).join(' '));
    }
}
var console = {
    log: function () { __wp_log('log', arguments); },
    info: function () { __wp_log('log', arguments); },
    warn: function () { __wp_log('warn', arguments); },
    error: function () { __wp_log('error', arguments); }
};
var shared = {};
var media = { available: false, playbackState: 0, title: '', artist: '', album: '', position: 0, duration: 0, artUrl: null };

// ── Script properties (the editor builder) ─────────────────────────────────

function createScriptProperties() {
    var api = {};
    function add(def) { if (def && def.name !== undefined) { api[def.name] = def.value; } return api; }
    ['addCheckbox', 'addSlider', 'addText', 'addTextInput', 'addCombo', 'addColor', 'addColorPicker',
        'addFile', 'addDirectory', 'addBool', 'addSoundFile'].forEach(function (m) {
        Object.defineProperty(api, m, { value: add, enumerable: false });
    });
    Object.defineProperty(api, 'finish', { value: function () { return api; }, enumerable: false });
    return api;
}

// ── engine / input / localStorage ──────────────────────────────────────────

function __wp_add_timer(cb, ms, repeat) {
    if (typeof cb !== 'function') { return 0; }
    var id = __wp.nextTimer++;
    var delay = Math.max(0, __wp_num(ms, 0)) / 1000;
    __wp.timers.push({ id: id, at: engine.runtime + delay, every: repeat ? Math.max(delay, 0.001) : 0, cb: cb });
    return id;
}
function __wp_clear_timer(id) {
    __wp.timers = __wp.timers.filter(function (t) { return t.id !== id; });
}

var engine = {
    runtime: 0,
    frametime: 0,
    timeOfDay: 0,
    canvasSize: new Vec2(1920, 1080),
    screenResolution: new Vec2(1920, 1080),
    userProperties: {},
    AUDIO_RESOLUTION_16: 16,
    AUDIO_RESOLUTION_32: 32,
    AUDIO_RESOLUTION_64: 64,
    registerAudioBuffers: function (resolution) {
        __wp.audioWanted = true;
        var n = (resolution === 16 || resolution === 32) ? resolution : 64;
        if (!__wp.audio) { __wp.audio = {}; }
        if (!__wp.audio[n]) {
            var z = function () { var a = []; for (var i = 0; i < n; i++) { a.push(0); } return a; };
            __wp.audio[n] = { resolution: n, left: z(), right: z(), average: z() };
        }
        return __wp.audio[n];
    },
    setTimeout: function (cb, ms) { return __wp_add_timer(cb, ms, false); },
    setInterval: function (cb, ms) { return __wp_add_timer(cb, ms, true); },
    clearTimeout: __wp_clear_timer,
    clearInterval: __wp_clear_timer,
    registerTimeEvent: function () {},
    isDesktopDevice: function () { return true; },
    isMobileDevice: function () { return false; },
    isWallpaper: function () { return true; },
    isScreensaver: function () { return false; },
    isRunningInEditor: function () { return false; },
    openUserShortcut: function (name) { __wp.commands.push(['shortcut', -1, String(name)]); }
};

var input = {
    cursorWorldPosition: new Vec3(0, 0, 0),
    cursorScreenPosition: new Vec2(0, 0),
    cursorLeftDown: false
};

var localStorage = {
    LOCATION_GLOBAL: 0,
    LOCATION_SCREEN: 1,
    get: function (key) { var v = __wp.storage[key]; return v === undefined ? undefined : JSON.parse(JSON.stringify(v)); },
    set: function (key, value) { __wp.storage[key] = value; __wp.storageDirty = true; },
    delete: function (key) { delete __wp.storage[key]; __wp.storageDirty = true; },
    remove: function (key) { delete __wp.storage[key]; __wp.storageDirty = true; },
    clear: function () { __wp.storage = {}; __wp.storageDirty = true; }
};

// ── Layer proxies ──────────────────────────────────────────────────────────
//
// Each scene object gets one persistent proxy. Vector-valued properties are
// returned as copies (as in WE, where they are engine-backed), so scripts
// must assign them back; every assignment marks the property dirty.

var __WP_VEC_PROPS = { origin: Vec3, angles: Vec3, scale: Vec3, size: Vec3, color: Vec3, parallaxDepth: Vec2 };

function __wp_clone(v) {
    if (v instanceof Vec2) { return new Vec2(v); }
    if (v instanceof Vec3) { return new Vec3(v); }
    if (v instanceof Vec4) { return new Vec4(v); }
    return v;
}

function __wp_coerce(prop, kind, v) {
    var C = __WP_VEC_PROPS[prop] || (kind === 'vec3' ? Vec3 : (kind === 'vec2' ? Vec2 : (kind === 'vec4' ? Vec4 : null)));
    if (C) { return new C(v); }
    if (kind === 'bool' || prop === 'visible') { return !!v; }
    if (kind === 'number' || prop === 'alpha' || prop === 'brightness' || prop === 'pointsize' || prop === 'volume') {
        var n = Number(v); return isFinite(n) ? n : 0;
    }
    if (kind === 'string' || prop === 'text') { return v === undefined || v === null ? '' : String(v); }
    return v;
}

function WPLayer(index, init) {
    this.__index = index;
    this.__s = {};
    this.__dirty = {};
    this.__kinds = {};
    this.__init = init.config || {};
    this.__frame = 0;
    this.__propagationDisabled = false;
    this.name = init.name || '';
    this.id = (init.id === null || init.id === undefined) ? -1 : init.id;
    this.debug = false;
    var s = this.__s;
    s.origin = new Vec3(init.origin);
    s.angles = new Vec3(init.angles);
    s.scale = new Vec3(init.scale === undefined ? '1 1 1' : init.scale);
    s.size = new Vec3(init.size);
    s.color = new Vec3(init.color === undefined ? '1 1 1' : init.color);
    s.parallaxDepth = new Vec2(init.parallaxDepth);
    s.visible = init.visible !== false;
    s.alpha = __wp_num(init.alpha, 1);
    s.brightness = __wp_num(init.brightness, 1);
    s.text = init.text === undefined ? '' : String(init.text);
    s.pointsize = __wp_num(init.pointsize, 32);
    s.font = init.font || '';
    s.volume = __wp_num(init.volume, 1);
    s.rate = __wp_num(init.rate, 1);
}

(function () {
    ['origin', 'angles', 'scale', 'size', 'color', 'parallaxDepth', 'visible', 'alpha', 'brightness',
        'text', 'pointsize', 'font', 'volume', 'rate'].forEach(function (p) {
        Object.defineProperty(WPLayer.prototype, p, {
            get: function () { return __wp_clone(this.__s[p]); },
            set: function (v) { this.__set(p, v); },
            enumerable: true
        });
    });
})();

WPLayer.prototype.__set = function (p, v) {
    var nv = __wp_coerce(p, this.__kinds[p], v);
    var ov = this.__s[p];
    var same = (nv !== null && typeof nv === 'object' && ov && typeof ov.equals === 'function') ? ov.equals(nv) : ov === nv;
    this.__s[p] = nv;
    if (!same) { this.__dirty[p] = true; }
};
WPLayer.prototype.__get = function (p) {
    if (p in this.__s) { return __wp_clone(this.__s[p]); }
    return undefined;
};
// Extra scriptable properties (shader constants, particle overrides…) are
// declared by Rust with an explicit kind and exposed under their full key.
WPLayer.prototype.__declare = function (p, kind, value) {
    this.__kinds[p] = kind;
    if (!(p in this.__s)) { this.__s[p] = __wp_coerce(p, kind, value); }
    if (!Object.getOwnPropertyDescriptor(WPLayer.prototype, p) && !Object.getOwnPropertyDescriptor(this, p)) {
        Object.defineProperty(this, p, {
            get: function () { return __wp_clone(this.__s[p]); },
            set: function (v) { this.__set(p, v); },
            enumerable: true, configurable: true
        });
    }
};

// Sound layers.
WPLayer.prototype.play = function () { __wp.commands.push(['sound', this.__index, 'play']); this.__playing = true; };
WPLayer.prototype.pause = function () { __wp.commands.push(['sound', this.__index, 'pause']); this.__playing = false; };
WPLayer.prototype.stop = function () { __wp.commands.push(['sound', this.__index, 'stop']); this.__playing = false; };
WPLayer.prototype.isPlaying = function () { return !!this.__playing; };

// Spritesheet animation.
WPLayer.prototype.getTextureAnimation = function () {
    var layer = this;
    return {
        setFrame: function (f) { layer.__frame = f | 0; layer.__dirty.__frame = true; },
        getFrame: function () { return layer.__frame; },
        getFrameCount: function () { return __wp_num(layer.__init.frameCount, 1); },
        setFrameByName: function () {},
        play: function () { __wp.commands.push(['texanim', layer.__index, 'play']); },
        pause: function () { __wp.commands.push(['texanim', layer.__index, 'pause']); },
        stop: function () { __wp.commands.push(['texanim', layer.__index, 'stop']); },
        isPlaying: function () { return true; },
        rate: 1
    };
};

// Property timeline animations (`"animation"` keyframes on a property).
WPLayer.prototype.getAnimation = function (name) {
    var layer = this, prop = String(name || '');
    var ctl = function (op) { return function () { __wp.commands.push(['anim', layer.__index, prop, op]); }; };
    return {
        play: ctl('play'), pause: ctl('pause'), stop: ctl('stop'),
        isPlaying: function () { return true; },
        setFrame: function (f) { __wp.commands.push(['anim', layer.__index, prop, 'frame', Number(f) || 0]); },
        getFrame: function () { return 0; },
        getFrameCount: function () { return 1; },
        rate: 1
    };
};

WPLayer.prototype.getParent = function () {
    var p = this.__init.parentIndex;
    return (typeof p === 'number' && p >= 0) ? (__wp.objects[p] || null) : null;
};
// Values in scene.json may be wrapped as {"value": …}; vectors arrive as
// "x y z" strings.
function __wp_raw_value(v) {
    v = __wp_value(v);
    if (typeof v === 'string') {
        var parts = v.trim().split(/\s+/);
        if (parts.length >= 2 && parts.length <= 4 && parts.every(function (p) { return p !== '' && isFinite(Number(p)); })) {
            return parts.length === 2 ? new Vec2(v) : (parts.length === 3 ? new Vec3(v) : new Vec4(v));
        }
    }
    return v;
}

// A Proxy whose properties read and write `layer.__s[prefix + name]`, falling
// back to `defaults` (raw scene.json values) until a script writes one.
function __wp_keyed_proxy(layer, prefix, defaults, fallback) {
    return new Proxy({}, {
        get: function (t, k) {
            if (typeof k !== 'string') { return undefined; }
            var key = prefix + k;
            if (key in layer.__s) { return __wp_clone(layer.__s[key]); }
            if (defaults && k in defaults) { return __wp_raw_value(defaults[k]); }
            return fallback;
        },
        set: function (t, k, v) { layer.__set(prefix + k, __wp_clone(v)); return true; },
        has: function (t, k) { return (prefix + k) in layer.__s || !!(defaults && k in defaults); }
    });
}

WPLayer.prototype.__effects = function () {
    var e = this.__init.raw && this.__init.raw.effects;
    return Array.isArray(e) ? e : [];
};
WPLayer.prototype.getEffectCount = function () { return this.__effects().length; };
WPLayer.prototype.getEffect = function (nameOrIndex) {
    var effects = this.__effects(), e = -1;
    if (typeof nameOrIndex === 'number') { e = nameOrIndex; }
    else {
        for (var i = 0; i < effects.length; i++) {
            var f = String(effects[i].file || '');
            if (effects[i].name === nameOrIndex || f.indexOf('/' + nameOrIndex + '/') >= 0) { e = i; break; }
        }
    }
    var raw = effects[e];
    if (!raw) { return null; }
    if (!this.__effectCache) { this.__effectCache = {}; }
    if (this.__effectCache[e]) { return this.__effectCache[e]; }
    var layer = this, passes = Array.isArray(raw.passes) ? raw.passes : [];
    var visKey = 'effects.' + e + '.visible';
    var fx = {
        name: raw.name || '',
        getMaterialCount: function () { return passes.length; },
        getMaterial: function (p) {
            p = p | 0;
            var consts = (passes[p] && passes[p].constantshadervalues) || {};
            return __wp_keyed_proxy(layer, 'effects.' + e + '.' + p + '.', consts, undefined);
        }
    };
    Object.defineProperty(fx, 'visible', {
        get: function () { return (visKey in layer.__s) ? layer.__s[visKey] : __wp_value(raw.visible) !== false; },
        set: function (v) { layer.__set(visKey, !!v); }
    });
    this.__effectCache[e] = fx;
    return fx;
};

// Particle instance controls (`instanceoverride` in scene.json).
Object.defineProperty(WPLayer.prototype, 'instance', {
    get: function () {
        if (!this.__instance) {
            var raw = (this.__init.raw && this.__init.raw.instanceoverride) || {};
            this.__instance = __wp_keyed_proxy(this, 'instance.', raw, 1);
        }
        return this.__instance;
    }
});

// Object-attachment API. Rotation composition follows the engine's own
// additive-Euler model (`render::Xform::compose`); there is no bone data, so
// attachment queries resolve to the object's own transform.
function __wp_rotate(v, a) {
    var r = WEMath.deg2rad;
    var sx = Math.sin(a.x * r), cx = Math.cos(a.x * r);
    var sy = Math.sin(a.y * r), cy = Math.cos(a.y * r);
    var sz = Math.sin(a.z * r), cz = Math.cos(a.z * r);
    var x1 = v.x, y1 = cx * v.y - sx * v.z, z1 = sx * v.y + cx * v.z;
    var x2 = cy * x1 + sy * z1, y2 = y1, z2 = -sy * x1 + cy * z1;
    return new Vec3(cz * x2 - sz * y2, sz * x2 + cz * y2, z2);
}
WPLayer.prototype.getTransformMatrix = function () {
    var a = this.__s.angles, s = this.__s.scale, o = this.__s.origin, r = WEMath.deg2rad;
    var sx = Math.sin(a.x * r), cx = Math.cos(a.x * r);
    var sy = Math.sin(a.y * r), cy = Math.cos(a.y * r);
    var sz = Math.sin(a.z * r), cz = Math.cos(a.z * r);
    return [
        cz * cy * s.x, sz * cy * s.x, -sy * s.x, 0,
        (cz * sy * sx - sz * cx) * s.y, (sz * sy * sx + cz * cx) * s.y, cy * sx * s.y, 0,
        (cz * sy * cx + sz * sx) * s.z, (sz * sy * cx - cz * sx) * s.z, cy * cx * s.z, 0,
        o.x, o.y, o.z, 1
    ];
};
WPLayer.prototype.lookAt = function (target) {
    var d = new Vec3(target).subtract(this.__s.origin), len = d.length(), a = this.angles;
    if (len > 1e-6) {
        a.y = Math.atan2(d.x, d.z) * WEMath.rad2deg;
        a.x = Math.asin(Math.max(-1, Math.min(1, -d.y / len))) * WEMath.rad2deg;
        this.angles = a;
    }
    return a;
};
WPLayer.prototype.lookAtYaw = function (target) {
    var d = new Vec3(target).subtract(this.__s.origin), a = this.angles;
    if (Math.abs(d.x) > 1e-6 || Math.abs(d.z) > 1e-6) { a.y = Math.atan2(d.x, d.z) * WEMath.rad2deg; this.angles = a; }
    return a;
};
WPLayer.prototype.rotateObjectSpace = function (x, y, z) {
    if (x !== null && typeof x === 'object') { z = x.z; y = x.y; x = x.x; }
    this.angles = this.angles.add(new Vec3(x, y, z));
    return this.angles;
};
WPLayer.prototype.setParent = function (parent) {
    if (!parent) { return this; }
    var ps = parent.scale;
    this.origin = __wp_rotate(this.origin.multiply(ps), parent.angles).add(parent.origin);
    this.angles = this.angles.add(parent.angles);
    this.scale = this.scale.multiply(ps);
    return this;
};
WPLayer.prototype.getAttachmentOrigin = function () { return this.origin; };
WPLayer.prototype.getAttachmentAngles = function () { return this.angles; };
WPLayer.prototype.getAttachmentMatrix = function () { return this.getTransformMatrix(); };
WPLayer.prototype.getAttachmentIndex = function () { return -1; };
WPLayer.prototype.disablepropagation = function () { this.__propagationDisabled = true; this.__dirty.__propagation = true; };

// ── thisScene ──────────────────────────────────────────────────────────────

// Scene settings default to WE's own defaults; `set_scene_general` copies
// scene.json's `general` values over them.
var thisScene = {
    bloom: false, bloomstrength: 2, bloomthreshold: 0.65,
    camerafade: true, cameraparallax: false, cameraparallaxamount: 0.5, cameraparallaxdelay: 0.1,
    cameraparallaxmouseinfluence: 0, camerashake: false, camerashakeamplitude: 0.5,
    camerashakeroughness: 1, camerashakespeed: 3, clearenabled: true,
    clearcolor: new Vec3(0, 0, 0), ambientcolor: new Vec3(0.2, 0.2, 0.2), skylightcolor: new Vec3(0.3, 0.3, 0.3),
    fov: 50, zoom: 1,
    getLayer: function (nameOrId) {
        var objs = __wp.objects;
        for (var i = 0; i < objs.length; i++) {
            if (objs[i] && (objs[i].name === nameOrId || objs[i].id === nameOrId)) { return objs[i]; }
        }
        return null;
    },
    getLayerCount: function () { return __wp.objects.filter(function (o) { return !!o; }).length; },
    enumerateLayers: function () { return __wp.objects.filter(function (o) { return !!o; }); },
    getLayerIndex: function (layer) { return layer ? __wp.objects.indexOf(layer) : -1; },
    sortLayer: function (layer, index) {
        if (layer) { __wp.commands.push(['sort', layer.__index, Number(index) || 0]); }
    },
    getInitialLayerConfig: function (layer) {
        return layer ? JSON.parse(JSON.stringify(layer.__init.raw || {})) : null;
    },
    createLayer: function (config) {
        var raw = (typeof config === 'string') ? { image: config } : JSON.parse(JSON.stringify(config || {}));
        var index = __wp.objects.length;
        var layer = new WPLayer(index, { name: raw.name, id: -1, origin: __wp_value(raw.origin), angles: __wp_value(raw.angles),
            scale: __wp_value(raw.scale), size: __wp_value(raw.size), visible: __wp_value(raw.visible),
            alpha: __wp_value(raw.alpha), color: __wp_value(raw.color), config: { raw: raw } });
        // Scene angles are radians; the proxy speaks degrees.
        layer.__s.angles = layer.__s.angles.multiply(WEMath.rad2deg);
        __wp.objects.push(layer);
        __wp.commands.push(['create', index, raw]);
        return layer;
    },
    destroyLayer: function (layer) {
        if (!layer) { return; }
        __wp.commands.push(['destroy', layer.__index]);
        __wp.objects[layer.__index] = null;
    }
};

function __wp_value(v) {
    return (v !== null && typeof v === 'object' && !(v instanceof Vec3) && 'value' in v) ? v.value : v;
}

// ── Script instances ───────────────────────────────────────────────────────

var __WP_EXPORTS = ['init', 'update', 'destroy', 'applyUserProperties', 'resizeScreen',
    'cursorClick', 'cursorDown', 'cursorUp', 'cursorMove', 'cursorEnter', 'cursorLeave',
    'mediaPlaybackChanged', 'mediaPropertiesChanged', 'mediaThumbnailChanged', 'mediaTimelineChanged',
    'mediaStatusChanged', 'animationEvent'];

function __wp_register_object(index, init) {
    var layer = new WPLayer(index, init);
    __wp.objects[index] = layer;
    return layer;
}

// `factory(thisLayer)` evaluates the script body once and returns its
// exported handlers; module-level state lives on in that closure.
function __wp_register_instance(index, prop, kind, authoredProps, factory) {
    var layer = __wp.objects[index];
    if (!layer) { return -1; }
    if (prop && kind) { layer.__declare(prop, kind, undefined); }
    var inst = { layer: layer, prop: prop, exports: null, inited: false, errors: 0, authored: authoredProps };
    try {
        inst.exports = factory(layer);
    } catch (e) {
        __wp_report(inst, 'load', e);
        inst.exports = {};
    }
    __wp.instances.push(inst);
    return __wp.instances.length - 1;
}

function __wp_report(inst, phase, e) {
    inst.errors++;
    if (inst.errors <= 3) {
        __wp_log('error', ['script on "' + inst.layer.name + '" (' + (inst.prop || 'events') + ') failed in ' + phase + ': ' + (e && e.message ? e.message : e)]);
    }
}

function __wp_call(inst, fn, args, phase) {
    var f = inst.exports && inst.exports[fn];
    if (typeof f !== 'function') { return undefined; }
    try { return f.apply(null, args || []); }
    catch (e) { __wp_report(inst, phase || fn, e); return undefined; }
}

function __wp_init_instance(inst) {
    inst.inited = true;
    var props = inst.exports.scriptProperties;
    if (props && inst.authored && typeof inst.authored === 'object') {
        for (var k in inst.authored) {
            var v = inst.authored[k];
            props[k] = (v !== null && typeof v === 'object' && 'value' in v) ? v.value : v;
        }
    }
    var cur = inst.prop ? inst.layer.__get(inst.prop) : undefined;
    var r = __wp_call(inst, 'init', [cur], 'init');
    if (r !== undefined && inst.prop) { inst.layer.__set(inst.prop, r); }
    __wp_call(inst, 'applyUserProperties', [engine.userProperties], 'applyUserProperties');
}

function __wp_frame(runtime, timeOfDay, frametime) {
    __wp.frame++;
    engine.runtime = runtime;
    engine.timeOfDay = timeOfDay;
    engine.frametime = frametime;

    // Timers due this frame (callbacks may add more; those wait a frame).
    var due = __wp.timers.filter(function (t) { return t.at <= runtime; });
    if (due.length) {
        __wp.timers = __wp.timers.filter(function (t) { return t.at > runtime || t.every > 0; });
        due.forEach(function (t) {
            if (t.every > 0) { t.at = runtime + t.every; }
            try { t.cb(); } catch (e) { __wp_log('error', ['timer callback failed: ' + (e && e.message ? e.message : e)]); }
        });
    }

    var list = __wp.instances;
    for (var i = 0; i < list.length; i++) {
        var inst = list[i];
        if (!inst.layer || __wp.objects[inst.layer.__index] !== inst.layer) { continue; }
        if (!inst.inited) { __wp_init_instance(inst); }
        if (inst.prop && inst.exports && typeof inst.exports.update === 'function') {
            var r = __wp_call(inst, 'update', [inst.layer.__get(inst.prop)], 'update');
            if (r !== undefined) { inst.layer.__set(inst.prop, r); }
        }
    }
}

// Run `fn(event)` on every instance attached to object `index` (or every
// instance when index < 0) — cursor and media events.
function __wp_dispatch(index, fn, event) {
    var list = __wp.instances;
    for (var i = 0; i < list.length; i++) {
        var inst = list[i];
        if (index >= 0 && inst.layer.__index !== index) { continue; }
        if (!inst.inited) { __wp_init_instance(inst); }
        __wp_call(inst, fn, [event], fn);
    }
}

function __wp_has_handler(fn) {
    return __wp.instances.some(function (inst) { return inst.exports && typeof inst.exports[fn] === 'function'; });
}

function __wp_handler_objects(fns) {
    var out = {};
    __wp.instances.forEach(function (inst) {
        if (inst.exports && fns.some(function (f) { return typeof inst.exports[f] === 'function'; })) {
            out[inst.layer.__index] = true;
        }
    });
    return Object.keys(out).map(Number);
}

function __wp_set_user_properties(props) {
    engine.userProperties = props;
    __wp.instances.forEach(function (inst) {
        if (inst.inited) { __wp_call(inst, 'applyUserProperties', [props], 'applyUserProperties'); }
    });
}

// Feed this frame's spectrum: arrays of 16/32/64 bands per channel.
function __wp_set_audio(l16, r16, l32, r32, l64, r64) {
    if (!__wp.audio) { return; }
    var src = { 16: [l16, r16], 32: [l32, r32], 64: [l64, r64] };
    for (var n in __wp.audio) {
        var b = __wp.audio[n], lr = src[n];
        for (var i = 0; i < b.resolution; i++) {
            var l = lr[0][i] || 0, r = lr[1][i] || 0;
            b.left[i] = l; b.right[i] = r; b.average[i] = (l + r) * 0.5;
        }
    }
}

function __wp_serialize(v) {
    if (v instanceof Vec2) { return [v.x, v.y]; }
    if (v instanceof Vec3) { return [v.x, v.y, v.z]; }
    if (v instanceof Vec4) { return [v.x, v.y, v.z, v.w]; }
    return v;
}

// Everything scripts changed since the last call, as one JSON string.
function __wp_collect() {
    var objects = [];
    for (var i = 0; i < __wp.objects.length; i++) {
        var o = __wp.objects[i];
        if (!o) { continue; }
        var keys = Object.keys(o.__dirty);
        if (!keys.length) { continue; }
        var d = {};
        keys.forEach(function (k) {
            if (k === '__frame') { d.frame = o.__frame; }
            else if (k === '__propagation') { d.propagationDisabled = o.__propagationDisabled; }
            else { d[k] = __wp_serialize(o.__s[k]); }
        });
        o.__dirty = {};
        objects.push([i, d]);
    }
    var out = { objects: objects, commands: __wp.commands, logs: __wp.logs };
    if (__wp.storageDirty) { out.storage = __wp.storage; __wp.storageDirty = false; }
    __wp.commands = [];
    __wp.logs = [];
    return JSON.stringify(out);
}
