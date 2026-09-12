// MangaVQ v1.6 luma encoder — точный порт _v15.js encodeFrame (cov+IBC-wide+pl8), pure std.
// IN:  [u32 nFrames][u32 Ke][Ke*8 codebook][per frame: u32 w, u32 h, w*h u8 luma]
// OUT: [per frame: u32 nParts=19][per part: u32 len + bytes(RAW, brotli на стороне node)][f64 sse][w*h u8 rec]
// Семантика JS отзеркалена: Math.round = floor(x+0.5); запись float в Uint8Array = усечение к нулю.
use std::fs::File;
use std::io::{Read, Write};
use std::thread;

const QD: f64 = 28.0;
const QE: i32 = 16;
const QT: i32 = 32;

#[inline] fn js_round(x: f64) -> i32 { (x + 0.5).floor() as i32 }
#[inline] fn clamp_i(v: i32) -> i32 { if v < 0 { 0 } else if v > 255 { 255 } else { v } }
#[inline] fn clamp_f(v: f64) -> f64 { if v < 0.0 { 0.0 } else if v > 255.0 { 255.0 } else { v } }
// ИНТЕГЕР-кост (2026-07-06): bits×2^20 в u64 — суммы ассоциативны (SIMD/порядок свободны), JS зеркалит
// теми же целыми в f64 (точно до 2^53). Сравнения ×0.9/×0.95 → ×10<×9 / ×20<×19 (целочисленно, точно).
const CSC: u64 = 1 << 20; // масштаб
static mut BITS_I: [u64; 512] = [0; 512];
fn init_bits_lut() { unsafe { for q in -255i32..=255 {
    let b = if q == 0 { 0.2 } else { 2.0 + (1.0 + q.abs() as f64).log2() };
    BITS_I[(q + 256) as usize] = (b * CSC as f64 + 0.5).floor() as u64;
} } }
#[inline] fn bits_i(q: i32) -> u64 { unsafe { BITS_I[(q + 256).clamp(0, 511) as usize] } }
// таблица кост-на-дифф для кванта ql: T[d+255] = BITS_I[js_round(d/ql)+256]
fn cost_tbl(ql: i32) -> Vec<u32> {
    let mut t = vec![0u32; 511];
    for d in -255i32..=255 { t[(d + 255) as usize] = bits_i(js_round(d as f64 / ql as f64)) as u32; }
    t
}

struct Plane { c: i32, a: i32, b: i32, me: f64 }
// специализация n=4 без клампов: cxy=1.5, Sxx=Syy=20; суммы целые (S2xv=Σ(2x-3)v точен),
// ga=js_round(S2xv/40.0) ≡ js_round(Sxv/20.0) (одно IEEE-деление эквивалентного значения) — бит-точно plane_fit
// специализация n=8 для внутренних блоков (клампы не нужны): целые суммы, me через ×2 (полуцелые точны)
fn plane_fit8_inner(y_plane: &[u8], w: usize, x0: usize, y0: usize) -> Plane {
    let mut isum = 0u32;
    for y in 0..8 { let row = &y_plane[(y0 + y) * w + x0..(y0 + y) * w + x0 + 8]; for &v in row { isum += v as u32; } }
    let c = js_round(isum as f64 / 64.0);
    let (mut s2xv, mut s2yv) = (0i32, 0i32);
    for y in 0..8 {
        let row = &y_plane[(y0 + y) * w + x0..(y0 + y) * w + x0 + 8];
        let dy2 = 2 * y as i32 - 7;
        for x in 0..8 { let v = row[x] as i32 - c; s2xv += (2 * x as i32 - 7) * v; s2yv += dy2 * v; }
    }
    let a = js_round(s2xv as f64 / 672.0);
    let b = js_round(s2yv as f64 / 672.0);
    // me в целых ×2: 2r = 2c + a(2x-7) + b(2y-7), clamp [0,510]; 2me целое → me = ime/2
    let mut ime = 0i32;
    for y in 0..8 {
        let row = &y_plane[(y0 + y) * w + x0..(y0 + y) * w + x0 + 8];
        let dy2 = 2 * y as i32 - 7;
        for x in 0..8 {
            let r2 = (2 * c + a * (2 * x as i32 - 7) + b * dy2).clamp(0, 510);
            let e = (r2 - 2 * row[x] as i32).abs(); if e > ime { ime = e; }
        }
    }
    Plane { c, a, b, me: ime as f64 / 2.0 }
}
fn plane_fit4(vv: &[u8; 16]) -> Plane {
    let mut isum = 0u32;
    for k in 0..16 { isum += vv[k] as u32; }
    let c = js_round(isum as f64 / 16.0);
    let (mut s2xv, mut s2yv) = (0i32, 0i32);
    for k in 0..16 {
        let v = vv[k] as i32 - c;
        let dx2 = 2 * (k & 3) as i32 - 3; let dy2 = 2 * (k >> 2) as i32 - 3;
        s2xv += dx2 * v; s2yv += dy2 * v;
    }
    let a = js_round(s2xv as f64 / 40.0);
    let b = js_round(s2yv as f64 / 40.0);
    let mut ime = 0i32;
    for k in 0..16 {
        let r2 = (2 * c + a * (2 * (k & 3) as i32 - 3) + b * (2 * (k >> 2) as i32 - 3)).clamp(0, 510);
        let e = (r2 - 2 * vv[k] as i32).abs(); if e > ime { ime = e; }
    }
    Plane { c, a, b, me: ime as f64 / 2.0 }
}
fn plane_fit<F: Fn(usize, usize) -> u8>(get: &F, n: usize) -> Plane {
    let cxy = (n as f64 - 1.0) / 2.0;
    let mut sum = 0.0;
    for y in 0..n { for x in 0..n { sum += get(x, y) as f64; } }
    let c = js_round(sum / (n * n) as f64);
    let (mut sxx, mut sxv, mut syy, mut syv) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..n { for x in 0..n {
        let v = get(x, y) as f64; let dx = x as f64 - cxy; let dy = y as f64 - cxy;
        sxx += dx * dx; sxv += dx * (v - c as f64); syy += dy * dy; syv += dy * (v - c as f64);
    }}
    let a = if sxx != 0.0 { js_round(sxv / sxx) } else { 0 };
    let b = if syy != 0.0 { js_round(syv / syy) } else { 0 };
    let mut me = 0.0f64;
    for y in 0..n { for x in 0..n {
        let r = clamp_f(c as f64 + a as f64 * (x as f64 - cxy) + b as f64 * (y as f64 - cxy));
        let e = (r - get(x, y) as f64).abs(); if e > me { me = e; }
    }}
    Plane { c, a, b, me }
}

fn dct8(bl: &[f64; 64], out: &mut [f64; 64], c8: &[[f64; 8]; 8]) {
    let mut t = [0.0f64; 64];
    for y in 0..8 { for u in 0..8 { let mut s = 0.0; for x in 0..8 { s += bl[y * 8 + x] * c8[u][x]; } t[y * 8 + u] = s; } }
    for u in 0..8 { for v in 0..8 { let mut s = 0.0; for y in 0..8 { s += t[y * 8 + u] * c8[v][y]; } out[v * 8 + u] = s; } }
}
fn idct8(co: &[f64; 64], out: &mut [f64; 64], c8: &[[f64; 8]; 8]) {
    let mut t = [0.0f64; 64];
    for v in 0..8 { for x in 0..8 { let mut s = 0.0; for u in 0..8 { s += co[v * 8 + u] * c8[u][x]; } t[v * 8 + x] = s; } }
    for x in 0..8 { for y in 0..8 { let mut s = 0.0; for v in 0..8 { s += t[v * 8 + x] * c8[v][y]; } out[y * 8 + x] = s; } }
}
// 4×4 DCT — НАДГРОБИЕ. Adaptive 8/4-DCT замерен в системе (_benchdct.js, blade QD=28) = МЁРТВ:
// вес +0.4..0.5% ХУЖЕ при любом λ (λ=0 потолок тоже), +14.7% энкод. Зонд обещал −12% но мерил
// ИЗОЛИРОВАННЫЙ DCT-поток; в системе DCT=24% люмы, доминирует rz-остаток (45%) который 4×4-рекон
// только ухудшает (соседи предсказывают от иного rec). НЕ включать. Оставлено для habr (measure-in-system).
#[allow(dead_code)]
fn c4_mat() -> &'static [[f64; 4]; 4] {
    static C: std::sync::OnceLock<[[f64; 4]; 4]> = std::sync::OnceLock::new();
    C.get_or_init(|| { let mut c = [[0.0f64; 4]; 4];
        for u in 0..4 { for x in 0..4 {
            c[u][x] = ((2.0 * x as f64 + 1.0) * u as f64 * std::f64::consts::PI / 8.0).cos() * if u == 0 { (1.0f64 / 4.0).sqrt() } else { (2.0f64 / 4.0).sqrt() };
        }} c })
}
#[allow(dead_code)]
fn dct4(bl: &[f64; 16], out: &mut [f64; 16], c4: &[[f64; 4]; 4]) {
    let mut t = [0.0f64; 16];
    for y in 0..4 { for u in 0..4 { let mut s = 0.0; for x in 0..4 { s += bl[y * 4 + x] * c4[u][x]; } t[y * 4 + u] = s; } }
    for u in 0..4 { for v in 0..4 { let mut s = 0.0; for y in 0..4 { s += t[y * 4 + u] * c4[v][y]; } out[v * 4 + u] = s; } }
}
#[allow(dead_code)]
fn idct4(co: &[f64; 16], out: &mut [f64; 16], c4: &[[f64; 4]; 4]) {
    let mut t = [0.0f64; 16];
    for v in 0..4 { for x in 0..4 { let mut s = 0.0; for u in 0..4 { s += co[v * 4 + u] * c4[u][x]; } t[v * 4 + x] = s; } }
    for x in 0..4 { for y in 0..4 { let mut s = 0.0; for v in 0..4 { s += t[v * 4 + x] * c4[v][y]; } out[y * 4 + x] = s; } }
}

// ---- SIMD-хелперы (интегер-кост ассоциативен → векторная сборка диффов безопасна для бит-точности) ----
#[cfg(target_arch = "x86_64")]
mod simd {
    #[inline] pub fn tier() -> u32 {
        use std::sync::OnceLock;
        static T: OnceLock<u32> = OnceLock::new();
        *T.get_or_init(|| {
            // MVQ_TIER=0/1/2 — форс (кламп к возможностям железа): валидация 512-ветки на Zen4 без JS-эталона
            let hw = if std::is_x86_feature_detected!("avx512bw") { 2 }
                else if std::is_x86_feature_detected!("avx2") { 1 }
                else { 0 };
            match std::env::var("MVQ_TIER").ok().and_then(|v| v.parse::<u32>().ok()) {
                Some(t) => t.min(hw),
                None => hw,
            }
        })
    }
    // 4x4: rec-окно и is_base по 4 строки; возвращает Some(диффы+255 как u16[16]) или None если не все base
    #[target_feature(enable = "avx2")]
    pub unsafe fn gather4_avx2(vv: &[u8; 16], rec: &[u8], is_base: &[u8], si0: usize, w: usize, out: &mut [u16; 16]) -> bool {
        use std::arch::x86_64::*;
        let mut rb = [0u8; 16]; let mut bb = [0u8; 16];
        for y in 0..4 {
            let o = si0 + y * w;
            rb[y * 4..y * 4 + 4].copy_from_slice(&rec[o..o + 4]);
            bb[y * 4..y * 4 + 4].copy_from_slice(&is_base[o..o + 4]);
        }
        let b = _mm_loadu_si128(bb.as_ptr() as *const __m128i);
        if _mm_movemask_epi8(_mm_cmpeq_epi8(b, _mm_set1_epi8(1))) != 0xFFFF { return false; }
        let v = _mm_loadu_si128(vv.as_ptr() as *const __m128i);
        let rc = _mm_loadu_si128(rb.as_ptr() as *const __m128i);
        let vlo = _mm256_cvtepu8_epi16(v); let rlo = _mm256_cvtepu8_epi16(rc);
        let d = _mm256_add_epi16(_mm256_sub_epi16(vlo, rlo), _mm256_set1_epi16(255));
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, d);
        true
    }
    // 8x8: то же для 64 пикселей
    #[target_feature(enable = "avx2")]
    pub unsafe fn gather8_avx2(vv: &[u8; 64], rec: &[u8], is_base: &[u8], si0: usize, w: usize, out: &mut [u16; 64]) -> bool {
        use std::arch::x86_64::*;
        let mut rb = [0u8; 64]; let mut bb = [0u8; 64];
        for y in 0..8 {
            let o = si0 + y * w;
            rb[y * 8..y * 8 + 8].copy_from_slice(&rec[o..o + 8]);
            bb[y * 8..y * 8 + 8].copy_from_slice(&is_base[o..o + 8]);
        }
        let ones = _mm256_set1_epi8(1);
        let b0 = _mm256_loadu_si256(bb.as_ptr() as *const __m256i);
        let b1 = _mm256_loadu_si256(bb.as_ptr().add(32) as *const __m256i);
        if _mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, ones)) != -1i32 { return false; }
        if _mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, ones)) != -1i32 { return false; }
        let k255 = _mm256_set1_epi16(255);
        for h in 0..4 {
            let v = _mm_loadu_si128(vv.as_ptr().add(h * 16) as *const __m128i);
            let rc = _mm_loadu_si128(rb.as_ptr().add(h * 16) as *const __m128i);
            let d = _mm256_add_epi16(_mm256_sub_epi16(_mm256_cvtepu8_epi16(v), _mm256_cvtepu8_epi16(rc)), k255);
            _mm256_storeu_si256(out.as_mut_ptr().add(h * 16) as *mut __m256i, d);
        }
        true
    }
    // AVX-512-ветка (НЕ валидирована на dev-железе Zen2 — исполнится только на машинах с avx512bw)
    #[target_feature(enable = "avx512bw", enable = "avx512vl")]
    pub unsafe fn gather8_avx512(vv: &[u8; 64], rec: &[u8], is_base: &[u8], si0: usize, w: usize, out: &mut [u16; 64]) -> bool {
        use std::arch::x86_64::*;
        let mut rb = [0u8; 64]; let mut bb = [0u8; 64];
        for y in 0..8 {
            let o = si0 + y * w;
            rb[y * 8..y * 8 + 8].copy_from_slice(&rec[o..o + 8]);
            bb[y * 8..y * 8 + 8].copy_from_slice(&is_base[o..o + 8]);
        }
        let b = _mm512_loadu_si512(bb.as_ptr() as *const __m512i);
        if _mm512_cmpeq_epi8_mask(b, _mm512_set1_epi8(1)) != u64::MAX { return false; }
        let v = _mm512_cvtepu8_epi16(_mm256_loadu_si256(vv.as_ptr() as *const __m256i));
        let rc = _mm512_cvtepu8_epi16(_mm256_loadu_si256(rb.as_ptr() as *const __m256i));
        let d0 = _mm512_add_epi16(_mm512_sub_epi16(v, rc), _mm512_set1_epi16(255));
        _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, d0);
        let v1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(vv.as_ptr().add(32) as *const __m256i));
        let rc1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(rb.as_ptr().add(32) as *const __m256i));
        let d1 = _mm512_add_epi16(_mm512_sub_epi16(v1, rc1), _mm512_set1_epi16(255));
        _mm512_storeu_si512(out.as_mut_ptr().add(32) as *mut __m512i, d1);
        true
    }
    // nn8-пара: обе 8-байтовые половины блока против кодбука за ОДИН проход (тот же порядок кандидатов)
    #[target_feature(enable = "avx2")]
    pub unsafe fn nn8_pair_avx2(cb: &[u8], ke: usize, vv: &[u8; 16]) -> (usize, usize) {
        use std::arch::x86_64::*;
        let v = _mm256_cvtepu8_epi16(_mm_loadu_si128(vv.as_ptr() as *const __m128i)); // [a:8|b:8] i16
        let (mut best_a, mut best_b) = (0usize, 0usize);
        let (mut bd_a, mut bd_b) = (i32::MAX, i32::MAX);
        let mut c = 0usize;
        while c + 2 <= ke {
            let cd0 = _mm_loadl_epi64(cb.as_ptr().add(c * 8) as *const __m128i);
            let cd1 = _mm_loadl_epi64(cb.as_ptr().add(c * 8 + 8) as *const __m128i);
            let cc0 = { let t = _mm_cvtepu8_epi16(cd0); _mm256_set_m128i(t, t) };
            let cc1 = { let t = _mm_cvtepu8_epi16(cd1); _mm256_set_m128i(t, t) };
            let d0 = _mm256_sub_epi16(v, cc0);
            let d1 = _mm256_sub_epi16(v, cc1);
            let sq0 = _mm256_madd_epi16(d0, d0);
            let sq1 = _mm256_madd_epi16(d1, d1);
            let h01 = _mm256_hadd_epi32(sq0, sq1);
            let h = _mm256_hadd_epi32(h01, h01);
            let lo = _mm256_castsi256_si128(h);
            let hi = _mm256_extracti128_si256(h, 1);
            // hadd 128-лейново: lo=[a(c), a(c+1),...], hi=[b(c), b(c+1),...]
            let da0 = _mm_cvtsi128_si32(lo); let da1 = _mm_extract_epi32(lo, 1);
            let db0 = _mm_cvtsi128_si32(hi); let db1 = _mm_extract_epi32(hi, 1);
            if da0 < bd_a { bd_a = da0; best_a = c; }
            if da1 < bd_a { bd_a = da1; best_a = c + 1; }
            if db0 < bd_b { bd_b = db0; best_b = c; }
            if db1 < bd_b { bd_b = db1; best_b = c + 1; }
            c += 2;
        }
        while c < ke {
            let (mut da, mut db) = (0i32, 0i32);
            for k in 0..8 { let e = vv[k] as i32 - cb[c * 8 + k] as i32; da += e * e; }
            for k in 0..8 { let e = vv[8 + k] as i32 - cb[c * 8 + k] as i32; db += e * e; }
            if da < bd_a { bd_a = da; best_a = c; }
            if db < bd_b { bd_b = db; best_b = c; }
            c += 1;
        }
        (best_a, best_b)
    }
    // nn8: SAD^2-поиск по кодбуку, 2 кандидата за такт AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn nn8_avx2(cb: &[u8], ke: usize, v: &[u8], off: usize) -> usize {
        use std::arch::x86_64::*;
        let vv = _mm_loadl_epi64(v.as_ptr().add(off) as *const __m128i); // 8 байт
        let v16 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(vv, vv));      // v|v как 16×i16
        let mut best = 0usize; let mut bd = i32::MAX;
        let mut c = 0usize;
        while c + 4 <= ke {
            let cc0 = _mm_loadu_si128(cb.as_ptr().add(c * 8) as *const __m128i);
            let cc1 = _mm_loadu_si128(cb.as_ptr().add(c * 8 + 16) as *const __m128i);
            let d0v = _mm256_sub_epi16(v16, _mm256_cvtepu8_epi16(cc0));
            let d1v = _mm256_sub_epi16(v16, _mm256_cvtepu8_epi16(cc1));
            let sq0 = _mm256_madd_epi16(d0v, d0v);
            let sq1 = _mm256_madd_epi16(d1v, d1v);
            // горизонтальные суммы 4 кандидатов разом: [c0a c0b c1a c1b ...]
            let h01 = _mm256_hadd_epi32(sq0, sq1); // [c0:2, c1:2 | c2:2, c3:2] (128-лейново)
            let h = _mm256_hadd_epi32(h01, h01);   // [c0, c1, c0, c1 | c2, c3, c2, c3]
            let lo = _mm256_castsi256_si128(h);
            let hi = _mm256_extracti128_si256(h, 1);
            // hadd 128-лейново: lo=[c0,c2..], hi=[c1,c3..]
            let d0 = _mm_cvtsi128_si32(lo); let d2 = _mm_extract_epi32(lo, 1);
            let d1 = _mm_cvtsi128_si32(hi); let d3 = _mm_extract_epi32(hi, 1);
            if d0 < bd { bd = d0; best = c; }
            if d1 < bd { bd = d1; best = c + 1; }
            if d2 < bd { bd = d2; best = c + 2; }
            if d3 < bd { bd = d3; best = c + 3; }
            c += 4;
        }
        while c < ke {
            let co = c * 8; let mut d = 0i32;
            for k in 0..8 { let e = v[off + k] as i32 - cb[co + k] as i32; d += e * e; }
            if d < bd { bd = d; best = c; }
            c += 1;
        }
        best
    }
}

fn nn8_pair(cb: &[u8], ke: usize, vv: &[u8; 16]) -> (usize, usize) {
    #[cfg(target_arch = "x86_64")]
    { if simd::tier() >= 1 { return unsafe { simd::nn8_pair_avx2(cb, ke, vv) }; } }
    (nn8(cb, ke, vv, 0), nn8(cb, ke, vv, 8))
}
fn nn8(cb: &[u8], ke: usize, v: &[u8], off: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    { if simd::tier() >= 1 { return unsafe { simd::nn8_avx2(cb, ke, v, off) }; } }
    let mut best = 0usize; let mut bd = i32::MAX;
    for c in 0..ke {
        let co = c * 8; let mut d = 0i32;
        for k in 0..8 { let e = v[off + k] as i32 - cb[co + k] as i32; d += e * e; }
        if d < bd { bd = d; best = c; }
    }
    best
}

// residCost с early-exit (интегер; limit=u64::MAX → без лимита)
fn resid_cost(vv: &[u8], base: &[u8], t: &[u32], limit: u64) -> u64 {
    let lim32 = if limit > u32::MAX as u64 { u32::MAX } else { limit as u32 };
    let mut cost = 0u32;
    for k in 0..vv.len() {
        cost += t[(vv[k] as i32 - base[k] as i32 + 255) as usize];
        if cost >= lim32 { return u64::MAX; }
    }
    cost as u64
}

struct Streams {
    m8: Vec<u8>, sm: Vec<u8>, pc: Vec<u8>, pa: Vec<u8>, pb: Vec<u8>,
    dsg: Vec<u8>, dmg: Vec<u16>, sub: Vec<u8>, s4: Vec<u8>,
    p4c: Vec<u8>, p4a: Vec<u8>, p4b: Vec<u8>, cov: Vec<u8>,
    ibcdx: Vec<u8>, ibcdy: Vec<u8>, it: Vec<u8>, ib: Vec<u8>,
    rz: [Vec<u8>; 4], qcls: Vec<u8>, imode: Vec<u8>,
}
// пресет3 класс3 = 8: тонкий класс (гладкое + ТЁМНОЕ ЗЕРНО — фикс «квадратов» 2026-07-06)
const QLS_PRESETS: [[i32; 4]; 6] = [[8,12,12,8],[10,14,14,10],[12,16,16,12],[14,18,18,10],[8,16,16,12],[8,18,18,14]];
#[inline] fn zig_push(rz: &mut [Vec<u8>; 4], q: i32, cls: usize) { let z = if q >= 0 { 2 * q } else { -2 * q - 1 }; rz[cls].push(z.min(255) as u8); }

// 4-класс Ql по занятости: [край, скринтон(aad>=6), зерно(aad>=1.2), гладкое]; таблица зависит от профиля страницы
const QLS_MANGA: [i32; 4] = [16, 32, 12, 8];
#[allow(dead_code)] const QLS_MANHWA: [i32; 4] = [8, 12, 12, 8];
fn qcls_of(vv: &[u8; 16], lo: i32, hi: i32) -> usize {
    if hi - lo >= 96 { return 0; }
    // int-гейты: aad>=6 ⟺ Σ|Δ|>=72 (72/12=6 точно); aad>=1.2 ⟺ Σ>=14.4 ⟺ Σ>=15 для целых — бит-точно f64-версии
    let mut s = 0i32;
    for y in 0..4 { for x in 0..3 { s += (vv[y * 4 + x] as i32 - vv[y * 4 + x + 1] as i32).abs(); } }
    if s >= 72 { 1 } else if s >= 15 { if hi <= 90 { 3 } else { 2 } } else { 3 } // тёмное зерно → тонкий класс
}

fn pack_bits(arr: &[u8]) -> Vec<u8> { let mut b = vec![0u8; (arr.len() + 7) >> 3]; for (i, &v) in arr.iter().enumerate() { if v != 0 { b[i >> 3] |= 1 << (i & 7); } } b }
fn pack2(arr: &[u8]) -> Vec<u8> { let mut b = vec![0u8; (arr.len() + 3) >> 2]; for (i, &v) in arr.iter().enumerate() { b[i >> 2] |= (v & 3) << ((i & 3) * 2); } b }
fn pack3(arr: &[u8]) -> Vec<u8> { let mut b = vec![0u8; (arr.len() * 3 + 7) / 8]; let mut bo = 0usize; for &v in arr { for t in 0..3 { if v & (1 << t) != 0 { b[bo >> 3] |= 1 << (bo & 7); } bo += 1; } } b }
fn u16le(arr: &[u16]) -> Vec<u8> { let mut b = Vec::with_capacity(arr.len() * 2); for &v in arr { b.extend_from_slice(&v.to_le_bytes()); } b }
// zero-split zig-остатков: битмаска ненулевых (прямой индекс по номеру значения; офсеты в vals = prefix-popcount)
// + ненулевые ниблы парами (15=эскейп-маркер) + эскейп-байты. RA-совместимо (GPU prefix-scan уже в архитектуре).
fn zsplit_pack(a: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut mask = vec![0u8; (a.len() + 7) / 8];
    let mut nz = Vec::with_capacity(a.len() / 3 + 1); let mut esc = Vec::new();
    let mut cur: i32 = -1;
    for (i, &v) in a.iter().enumerate() {
        if v == 0 { continue; }
        mask[i >> 3] |= 1 << (i & 7);
        let x = if v < 15 { v } else { esc.push(v); 15 };
        if cur < 0 { cur = x as i32; } else { nz.push((cur as u8) | (x << 4)); cur = -1; }
    }
    if cur >= 0 { nz.push(cur as u8); }
    (mask, nz, esc)
}
// dmg (знаковые DCT-кванты u16→i16): zig → zero-split; z>=255 → эскейп 255 + 2 байта BE
fn zsplit_dmg(a: &[u16]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut mask = vec![0u8; (a.len() + 7) / 8];
    let mut nz = Vec::with_capacity(a.len() / 3 + 1); let mut esc = Vec::new();
    let mut cur: i32 = -1;
    for (i, &w) in a.iter().enumerate() {
        let v = w as i16 as i32;
        let z = if v >= 0 { 2 * v } else { -2 * v - 1 };
        if z == 0 { continue; }
        mask[i >> 3] |= 1 << (i & 7);
        let x = if z < 15 { z as u8 } else {
            if z < 255 { esc.push(z as u8); } else { esc.push(255); esc.push((z >> 8) as u8); esc.push((z & 255) as u8); }
            15
        };
        if cur < 0 { cur = x as i32; } else { nz.push((cur as u8) | (x << 4)); cur = -1; }
    }
    if cur >= 0 { nz.push(cur as u8); }
    (mask, nz, esc)
}

fn encode_frame(y_plane: &[u8], w: usize, h: usize, cb: &[u8], ke: usize, c8: &[[f64; 8]; 8], mh: bool, mhdct: i32, qpreset: usize, fast: bool, m8c: Option<&[u8]>) -> (Vec<Vec<u8>>, f64, Vec<u8>) {
    let prof = std::env::var("MVQ_PROF").is_ok();
    let t_start = std::time::Instant::now();
    let (mut t_det, mut t_dct, mut t_ibc8, mut t_pl8, mut t_nn, mut t_ibc4, mut t_intra) = (0u128,0u128,0u128,0u128,0u128,0u128,0u128);
    let qls: &[i32; 4] = if mh { &QLS_PRESETS[qpreset.min(5)] } else { &QLS_MANGA };
    let ct: [Vec<u32>; 4] = [cost_tbl(qls[0]), cost_tbl(qls[1]), cost_tbl(qls[2]), cost_tbl(qls[3])];
    let ct_qt: Vec<u32> = cost_tbl(QT);
    let qde: f64 = if mh && mhdct > 0 { mhdct as f64 } else { QD };
    // line-aware маршрутизация — стандарт mh-профиля (зеркало JS: Float32Array-хранение box3)
    let lq = mh;
    let mut line_mask: Vec<u8> = Vec::new();
    let __t_det = std::time::Instant::now();
    if lq {
        // ЦЕЛОЧИСЛЕННЫЙ box3×box3-детектор (общий знаменатель 420=lcm(4..7)): деления заменены кросс-умножением.
        // Семантика синхронно сменена в JS (_v15) 2026-07-06 — бит-точность сохранена, f32-округления t убраны.
        // T'[x] = sx*(420/nx); маска: 420*ny*Y < S - 7560*ny && Y < 160, где S = Σ_dy T'
        // zip-итераторы вместо индексаций: LLVM снимает bounds-checks и векторизует
        let mut tprime = vec![0i32; w * h];
        let mut acc = vec![0i32; w];
        for y in 0..h {
            for v in acc.iter_mut() { *v = 0; }
            let row = &y_plane[y * w..y * w + w];
            for d in -3i32..=3 {
                if d < 0 {
                    let dd = (-d) as usize;
                    for (a, &v) in acc[dd..].iter_mut().zip(row[..w - dd].iter()) { *a += v as i32; }
                } else {
                    let dd = d as usize;
                    for (a, &v) in acc[..w - dd].iter_mut().zip(row[dd..].iter()) { *a += v as i32; }
                }
            }
            let trow = &mut tprime[y * w..y * w + w];
            for (t, &a) in trow[3..w - 3].iter_mut().zip(acc[3..w - 3].iter()) { *t = a * 60; } // внутренние: nx=7
            for x in 0..3.min(w) {
                let nx = ((x + 3).min(w - 1) + 1) as i32;
                trow[x] = acc[x] * (420 / nx);
            }
            for x in w.saturating_sub(3)..w {
                let nx = ((x + 3).min(w - 1) - x.saturating_sub(3) + 1) as i32;
                trow[x] = acc[x] * (420 / nx);
            }
        }
        line_mask = vec![0u8; w * h];
        let mut sacc = vec![0i32; w];
        for y in 0..h {
            for v in sacc.iter_mut() { *v = 0; }
            let mut ny = 0i32;
            for d in -3i32..=3 {
                let yy = y as i32 + d; if yy < 0 || yy as usize >= h { continue; }
                ny += 1;
                let tr = &tprime[yy as usize * w..yy as usize * w + w];
                for (s, &t) in sacc.iter_mut().zip(tr.iter()) { *s += t; }
            }
            let (thr_mul, thr_sub) = (420 * ny, 7560 * ny);
            let row = &y_plane[y * w..y * w + w];
            let mrow = &mut line_mask[y * w..y * w + w];
            for ((m, &yv8), &s) in mrow.iter_mut().zip(row.iter()).zip(sacc.iter()) {
                let yv = yv8 as i32;
                *m = ((yv < 160) & (thr_mul * yv < s - thr_sub)) as u8;
            }
        }
    }
    t_det = __t_det.elapsed().as_nanos();
    let has_line = |x0: usize, y0: usize, n: usize| -> bool {
        if !lq { return false; }
        for y in 0..n { for x in 0..n { let xx = x0 + x; let yy = y0 + y; if xx < w && yy < h && line_mask[yy * w + xx] == 1 { return true; } } }
        false
    };
    let mut rec = vec![0u8; w * h];
    let mut is_base = vec![0u8; w * h];
    let bw4 = (w + 3) / 4; let bh4 = (h + 3) / 4;
    let mut offs4: Vec<(i32,i32)> = vec![(0,0); bw4 * bh4]; // выбранные IBC-офсеты для предикторов
    let bw8g = (w + 7) / 8; let mut offs8: Vec<(i32,i32)> = vec![(0,0); bw8g * ((h + 7) / 8)];
    let bw8 = (w + 7) / 8; let bh8 = (h + 7) / 8;
    let cap = w * h / 8;
    let mut s = Streams { m8: vec![], sm: vec![], pc: vec![], pa: vec![], pb: vec![], dsg: vec![], dmg: vec![], sub: vec![], s4: vec![], p4c: vec![], p4a: vec![], p4b: vec![], cov: vec![], ibcdx: vec![], ibcdy: vec![], it: vec![], ib: vec![], rz: [Vec::with_capacity(cap), Vec::with_capacity(cap), Vec::with_capacity(cap), Vec::with_capacity(cap)], qcls: vec![], imode: vec![] };
    let mut sse = 0.0f64;
    let getp = |xx: usize, yy: usize| -> u8 { y_plane[yy.min(h - 1) * w + xx.min(w - 1)] };

    for by in 0..bh8 { for bx in 0..bw8 {
        let x0 = bx * 8; let y0 = by * 8;
        let get = |x: usize, y: usize| -> u8 { getp(x0 + x, y0 + y) };
        // ---- mode8 классификация (кэш из cb-скана, если есть; математика идентична — вес байт-в-байт) ----
        let mcls: Option<u8> = m8c.and_then(|m| m.get(by * bw8 + bx).copied()).filter(|&c| c <= 3);
        let (lo, hi, sum, sd, ext);
        match mcls {
            Some(0) => { let mut s0 = 0f64; for y in 0..8 { for x in 0..8 { s0 += get(x, y) as f64; } } lo = 0; hi = 2; sum = s0; sd = 0.0; ext = 64; }
            Some(1) => { lo = 0; hi = 255; sum = 0.0; sd = 0.0; ext = 64; }
            Some(2) => { lo = 0; hi = 255; sum = 0.0; sd = 22.0; ext = 0; }
            Some(_) => { lo = 0; hi = 255; sum = 0.0; sd = 0.0; ext = 64; }
            None => {
                let (mut l2, mut h2, mut s2) = (255i32, 0i32, 0f64);
                for y in 0..8 { for x in 0..8 { let v = get(x, y) as i32; if v < l2 { l2 = v; } if v > h2 { h2 = v; } s2 += v as f64; } }
                let mean2 = s2 / 64.0;
                let (mut vr2, mut e2) = (0.0f64, 0i32);
                for y in 0..8 { for x in 0..8 { let v = get(x, y) as i32; vr2 += (v as f64 - mean2) * (v as f64 - mean2); if v < l2 + 40 || v > h2 - 40 { e2 += 1; } } }
                lo = l2; hi = h2; sum = s2; sd = (vr2 / 64.0).sqrt(); ext = e2;
            }
        }
        let mean = sum / 64.0;
        let mut mcode: u8;
        if hi - lo <= 2 {
            mcode = 0;
            let m = js_round(mean);
            s.sm.push(m as u8);
            for y in 0..8 { for x in 0..8 {
                let (px, py) = (x0 + x, y0 + y);
                if px < w && py < h { rec[py * w + px] = m as u8; is_base[py * w + px] = 1; }
                let e = m as f64 - get(x, y) as f64; sse += e * e;
            }}
        } else {
            let need_pf = mcls.map_or(true, |c| c == 1);
            let pf = if need_pf {
                if x0 + 8 <= w && y0 + 8 <= h { plane_fit8_inner(y_plane, w, x0, y0) } else { plane_fit(&get, 8) }
            } else { Plane { c: 0, a: 0, b: 0, me: 255.0 } };
            // None-путь повторяет mode8_of-гейт (кэш Some(1) уже прошёл его в mode8_of)
            // перестройка критерия (2026-07-08): тёмное исключаем из planar8 по ЗЕРНУ (planar плохо описывает),
            // НЕ по яркости. Гладкий тёмный градиент (pf.me низкий) → planar8 (однородно, без split4-мозаики).
            let dark_grain_blk = mcls.is_none() && hi <= 90 && sd >= 1.2 && pf.me > pl8_darkres();
            if pf.me <= 6.0 && !dark_grain_blk {
                mcode = 1;
                s.pc.push((pf.c & 255) as u8); s.pa.push((pf.a & 255) as u8); s.pb.push((pf.b & 255) as u8);
                for y in 0..8 { for x in 0..8 {
                    let r = clamp_f(pf.c as f64 + pf.a as f64 * (x as f64 - 3.5) + pf.b as f64 * (y as f64 - 3.5));
                    let (px, py) = (x0 + x, y0 + y);
                    if px < w && py < h { rec[py * w + px] = r as u8; is_base[py * w + px] = 1; }
                    let e = r - get(x, y) as f64; sse += e * e;
                }}
            } else if (!mh || mhdct > 0) && !(lq && has_line(x0, y0, 8)) && sd >= 22.0 && sd <= 115.0 && (ext as f64) / 64.0 < 0.72 {
                // DCT-скринтон + IBC8-кандидат
                mcode = 2;
                let __t_dct = std::time::Instant::now();
                let mut bl = [0.0f64; 64];
                for y in 0..8 { for x in 0..8 { bl[y * 8 + x] = get(x, y) as f64; } }
                let mut co = [0.0f64; 64]; dct8(&bl, &mut co, c8);
                let mut qs = [0i32; 64]; let mut qd = [0.0f64; 64]; let mut dcost = 0u64;
                for k in 0..64 { let v = js_round(co[k] / qde); qs[k] = v; qd[k] = v as f64 * qde; dcost += bits_i(v); }
                let mut rc = [0.0f64; 64]; idct8(&qd, &mut rc, c8);
                let mut drec = [0u8; 64]; let mut dsse = 0.0; let mut k2 = 0;
                for y in 0..8 { for x in 0..8 { let r = clamp_i(js_round(rc[k2])); drec[k2] = r as u8; let e = r as f64 - get(x, y) as f64; dsse += e * e; k2 += 1; } }
                t_dct += __t_dct.elapsed().as_nanos();
                // IBC8
                let mut vv = [0u8; 64]; let mut k3 = 0;
                for y in 0..8 { for x in 0..8 { vv[k3] = get(x, y); k3 += 1; } }
                let mut best_cost = u64::MAX; let (mut bdx, mut bdy) = (0i32, 0i32);
                let __t_i8 = std::time::Instant::now();
                {
                    let g8x = x0 / 8; let g8y = y0 / 8;
                    let mut tried = [0u64; 32]; // битсет 2016 офсетов: idx = dy*63+dx+31
                    macro_rules! eval8 { ($dx:expr,$dy:expr) => {{
                        let dx = $dx; let dy = $dy;
                        let valid = !(dy < 0 || (dy == 0 && dx <= 0) || dx < -31 || dx > 31 || dy > 31);
                        let ti = if valid { (dy * 63 + dx + 31) as usize } else { 0 };
                        if valid && (!fast || tried[ti >> 6] & (1u64 << (ti & 63)) == 0) {
                            if fast { tried[ti >> 6] |= 1u64 << (ti & 63); }
                            let sx0 = x0 as i32 - dx; let sy0 = y0 as i32 - dy;
                            if sx0 >= 0 && sy0 >= 0 && sx0 as usize + 8 <= w && sy0 as usize + 8 <= h {
                                let limit = best_cost.saturating_sub(10 * CSC);
                                let lim32 = if limit > u32::MAX as u64 { u32::MAX } else { limit as u32 };
                                let si0 = sy0 as usize * w + sx0 as usize;
                                let mut ok;
                                let mut cost = 0u32;
                                #[cfg(target_arch = "x86_64")]
                                {
                                    let t = simd::tier();
                                    if t >= 1 {
                                        let mut dif = [0u16; 64];
                                        ok = unsafe { if t >= 2 { simd::gather8_avx512(&vv, &rec, &is_base, si0, w, &mut dif) }
                                                      else { simd::gather8_avx2(&vv, &rec, &is_base, si0, w, &mut dif) } };
                                        if ok {
                                            for h2 in 0..4 {
                                                for k in h2*16..h2*16+16 { cost += ct_qt[dif[k] as usize]; }
                                                if cost >= lim32 { ok = false; break; }
                                            }
                                        }
                                    } else {
                                        ok = true; let mut k4 = 0usize;
                                        'o8s: for y in 0..8usize { for x in 0..8usize {
                                            let si = si0 + y * w + x;
                                            if is_base[si] == 0 { ok = false; break 'o8s; }
                                            cost += ct_qt[(vv[k4] as i32 - rec[si] as i32 + 255) as usize];
                                            if cost >= lim32 { ok = false; break 'o8s; }
                                            k4 += 1;
                                        }}
                                    }
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    ok = true; let mut k4 = 0usize;
                                    'o8s: for y in 0..8usize { for x in 0..8usize {
                                        let si = si0 + y * w + x;
                                        if is_base[si] == 0 { ok = false; break 'o8s; }
                                        cost += ct_qt[(vv[k4] as i32 - rec[si] as i32 + 255) as usize];
                                        if cost >= limit { ok = false; break 'o8s; }
                                        k4 += 1;
                                    }}
                                }
                                if ok { let cost = cost as u64 + 10 * CSC; if cost < best_cost { best_cost = cost; bdx = dx; bdy = dy; } }
                            }
                        }
                    }} }
                    if fast {
                    // DCT-сиды периода скринтона (пик AC по строке/столбцу спектра)
                    {
                        let mut kx = 1usize; let mut mv = 0.0f64;
                        for u in 1..8 { let a = co[u].abs(); if a > mv { mv = a; kx = u; } }
                        let mut ky = 1usize; mv = 0.0;
                        for v in 1..8 { let a = co[v*8].abs(); if a > mv { mv = a; ky = v; } }
                        let px = js_round(16.0 / kx as f64); let py = js_round(16.0 / ky as f64);
                        eval8!(px, 0); eval8!(0, py); eval8!(px, py); eval8!(px*2, 0); eval8!(0, py*2);
                    }
                    if g8x > 0 { let (px,py) = offs8[g8y*bw8g+g8x-1]; if px!=0||py!=0 { eval8!(px,py); } }
                    if g8y > 0 { let (px,py) = offs8[(g8y-1)*bw8g+g8x]; if px!=0||py!=0 { eval8!(px,py); } }
                    if g8x > 0 && g8y > 0 { let (px,py) = offs8[(g8y-1)*bw8g+g8x-1]; if px!=0||py!=0 { eval8!(px,py); } }
                    if g8y > 0 && g8x+1 < bw8g { let (px,py) = offs8[(g8y-1)*bw8g+g8x+1]; if px!=0||py!=0 { eval8!(px,py); } }
                    // early-accept: предиктор почти идеален (~нулевой остаток) → окно/refine не нужны
                    if best_cost > 26 * CSC {
                    for dy in 0..=8i32 { let dxs = if dy>0 {-8} else {1}; for dx in dxs..=8i32 { eval8!(dx,dy); } }
                    for _round in 0..2 { let (bx2,by2)=(bdx,bdy); if best_cost != u64::MAX {
                        for dy in -2i32..=2 { for dx in -2i32..=2 { if dx==0&&dy==0 {continue;} eval8!(bx2+dx,by2+dy); } } } }
                    }
                    } else {
                    // полный перебор — дефолт (вес важнее; fast=бит19 = опциональный скоростной режим)
                    for dy in 0..=31i32 { let dxs = if dy>0 {-31} else {1}; for dx in dxs..=31i32 { eval8!(dx,dy); } }
                    }
                }
                t_ibc8 += __t_i8.elapsed().as_nanos();
                if best_cost.saturating_mul(10) < dcost.saturating_mul(9) {
                    mcode = 4;
                    offs8[(y0/8)*bw8g + x0/8] = (bdx, bdy);
                    s.ibcdx.push((bdx & 0xff) as u8); s.ibcdy.push((bdy & 0xff) as u8);
                    let (sx0, sy0) = ((x0 as i32 - bdx) as usize, (y0 as i32 - bdy) as usize);
                    for k in 0..64 {
                        let si = (sy0 + (k >> 3)) * w + sx0 + (k & 7);
                        let bv = rec[si];
                        let q = js_round((vv[k] as f64 - bv as f64) / QT as f64);
                        zig_push(&mut s.rz, q, 1);
                        let r = clamp_i(bv as i32 + q * QT);
                        let (px, py) = (x0 + (k & 7), y0 + (k >> 3));
                        if px < w && py < h { rec[py * w + px] = r as u8; }
                        let e = r as f64 - vv[k] as f64; sse += e * e;
                    }
                } else {
                    for k in 0..64 { s.dsg.push(if qs[k] < 0 { 1 } else { 0 }); s.dmg.push(qs[k].abs().min(65535) as u16); }
                    let mut k7 = 0;
                    for y in 0..8 { for x in 0..8 { let (px, py) = (x0 + x, y0 + y); if px < w && py < h { rec[py * w + px] = drec[k7]; is_base[py * w + px] = 1; } k7 += 1; } }
                    sse += dsse;
                }
            } else {
                // pl8-кандидат перед split4
                mcode = 3;
                let mut pl8_done = false;
                let __t_pl = std::time::Instant::now();
                {
                    let (mut lo8, mut hi8, mut sum8) = (255i32, 0i32, 0f64);
                    for y in 0..8 { for x in 0..8 { let v = get(x, y) as i32; if v < lo8 { lo8 = v; } if v > hi8 { hi8 = v; } sum8 += v as f64; } }
                    let mean8 = sum8 / 64.0; let mut vr8 = 0.0;
                    for y in 0..8 { for x in 0..8 { let v = get(x, y) as f64; vr8 += (v - mean8) * (v - mean8); } }
                    let sd8 = (vr8 / 64.0).sqrt();
                    if sd8 < 14.0 && !(lq && has_line(x0, y0, 8)) {
                        let pf8 = if x0 + 8 <= w && y0 + 8 <= h { plane_fit8_inner(y_plane, w, x0, y0) } else { plane_fit(&get, 8) };
                        let mut aad8 = 0.0f64; let mut na8 = 0i32;
                        for y in 0..8 { for x in 0..7 { aad8 += (get(x, y) as f64 - get(x + 1, y) as f64).abs(); na8 += 1; } }
                        aad8 /= na8 as f64;
                        let cls8: usize = if hi8 - lo8 >= 96 { 0 } else if aad8 >= 6.0 { 1 } else if aad8 >= 1.2 { if hi8 <= 90 { 3 } else { 2 } } else { 3 };
                        let ql8 = qls[cls8];
                        let mut cost_pl8 = 26 * CSC; let mut pl8q = [0i32; 64]; let mut k = 0; let mut pl8res = 0.0f64;
                        for y in 0..8 { for x in 0..8 {
                            let basef = clamp_f(pf8.c as f64 + pf8.a as f64 * (x as f64 - 3.5) + pf8.b as f64 * (y as f64 - 3.5));
                            pl8res += (get(x, y) as f64 - basef).abs();
                            let q = js_round((get(x, y) as f64 - basef) / ql8 as f64);
                            pl8q[k] = q; k += 1; cost_pl8 += bits_i(q);
                        }}
                        let pl8_me = pl8res / 64.0; // средний planar-остаток: мал=градиент(→pl8), велик=зерно(→split4)
                        // дешёвая VQ-аппроксимация split4 (как в JS: solid/planar/VQ, без ibc/cov)
                        let mut cost_vq4 = 0u64;
                        for sy2 in 0..2 { for sx2 in 0..2 {
                            let mut vv = [0u8; 16];
                            if x0 + sx2 * 4 + 4 <= w && y0 + sy2 * 4 + 4 <= h {
                                let base = (y0 + sy2 * 4) * w + x0 + sx2 * 4;
                                for row in 0..4 { vv[row * 4..row * 4 + 4].copy_from_slice(&y_plane[base + row * w..base + row * w + 4]); }
                            } else {
                                for k2 in 0..16 { vv[k2] = getp(x0 + sx2 * 4 + (k2 & 3), y0 + sy2 * 4 + (k2 >> 2)); }
                            }
                            let (mut lo, mut hi) = (255i32, 0i32);
                            for k2 in 0..16 { let v = vv[k2] as i32; if v < lo { lo = v; } if v > hi { hi = v; } }
                            if hi - lo <= 2 { cost_vq4 += 10 * CSC; continue; }
                            let pf = plane_fit4(&vv);
                            if pf.me <= 6.0 { cost_vq4 += 26 * CSC; continue; }
                            let (a, b) = nn8_pair(cb, ke, &vv);
                            let qi = qcls_of(&vv, lo, hi);
                            let mut vbase = [0u8; 16];
                            for k2 in 0..16 { vbase[k2] = if k2 < 8 { cb[a * 8 + k2] } else { cb[b * 8 + (k2 - 8)] }; }
                            cost_vq4 += resid_cost(&vv, &vbase, &ct[qi], u64::MAX) + 16 * CSC;
                        }}
                        // тёмное зерно НЕ в pl8 (гейт по РЕАЛЬНЫМ hi8/aad8 — кэш-фиктивные hi/sd не годятся): split4-IBC скопирует паттерн
                        if !(hi8 <= 90 && aad8 >= 1.2 && pl8_me > pl8_darkres()) && cost_pl8.saturating_mul(20) < cost_vq4.saturating_mul(19) {
                            s.m8.push(5); s.qcls.push(cls8 as u8);
                            s.pc.push((pf8.c & 255) as u8); s.pa.push((pf8.a & 255) as u8); s.pb.push((pf8.b & 255) as u8);
                            let mut k3 = 0;
                            for y in 0..8 { for x in 0..8 {
                                let basef = clamp_f(pf8.c as f64 + pf8.a as f64 * (x as f64 - 3.5) + pf8.b as f64 * (y as f64 - 3.5));
                                let q = pl8q[k3]; k3 += 1;
                                zig_push(&mut s.rz, q, cls8);
                                let r = clamp_f(basef + (q * ql8) as f64);
                                let (px, py) = (x0 + x, y0 + y);
                                if px < w && py < h { rec[py * w + px] = r as u8; is_base[py * w + px] = 1; }
                                let e = r - get(x, y) as f64; sse += e * e;
                            }}
                            pl8_done = true;
                        }
                    }
                }
                t_pl8 += __t_pl.elapsed().as_nanos();
                if pl8_done { continue; }
                // ---- split4 ----
                for sy2 in 0..2usize { for sx2 in 0..2usize {
                    let ox = x0 + sx2 * 4; let oy = y0 + sy2 * 4;
                    let mut vv = [0u8; 16];
                    if ox + 4 <= w && oy + 4 <= h {
                        let base = oy * w + ox;
                        for row in 0..4 { vv[row * 4..row * 4 + 4].copy_from_slice(&y_plane[base + row * w..base + row * w + 4]); }
                    } else {
                        for k in 0..16 { vv[k] = getp(ox + (k & 3), oy + (k >> 2)); }
                    }
                    let (mut lo, mut hi) = (255i32, 0i32);
                    for k in 0..16 { let v = vv[k] as i32; if v < lo { lo = v; } if v > hi { hi = v; } }
                    if hi - lo <= 2 {
                        s.sub.push(0);
                        let mm = js_round((lo + hi) as f64 / 2.0);
                        s.s4.push(mm as u8);
                        for k in 0..16 { let (px, py) = (ox + (k & 3), oy + (k >> 2)); if px < w && py < h { rec[py * w + px] = mm as u8; is_base[py * w + px] = 1; } let e = mm as f64 - vv[k] as f64; sse += e * e; }
                        continue;
                    }
                    let pf = plane_fit4(&vv);
                    if pf.me <= 6.0 {
                        s.sub.push(1);
                        s.p4c.push((pf.c & 255) as u8); s.p4a.push((pf.a & 255) as u8); s.p4b.push((pf.b & 255) as u8);
                        for k in 0..16 {
                            let r = clamp_f(pf.c as f64 + pf.a as f64 * ((k & 3) as f64 - 1.5) + pf.b as f64 * ((k >> 2) as f64 - 1.5));
                            let (px, py) = (ox + (k & 3), oy + (k >> 2)); if px < w && py < h { rec[py * w + px] = r as u8; is_base[py * w + px] = 1; }
                            let e = r - vv[k] as f64; sse += e * e;
                        }
                        continue;
                    }
                    // covOK
                    let cov_ok = if mh { false } else {
                        let (mut sl, mut nl, mut dark, mut mn, mut mx) = (0f64, 0i32, 0i32, 255i32, 0i32);
                        for k in 0..16 { let v = vv[k] as i32; if v >= 200 { sl += v as f64; nl += 1; } if v < 170 { dark += 1; } if v < mn { mn = v; } if v > mx { mx = v; } }
                        if mx - mn <= 3 && mn >= 236 { true }
                        else if nl >= 4 {
                            let m = sl / nl as f64; let (mut sdv, mut c2) = (0f64, 0i32);
                            for k in 0..16 { if vv[k] as i32 >= 200 { let d = vv[k] as f64 - m; sdv += d * d; c2 += 1; } }
                            let sdc = (sdv / c2 as f64).sqrt();
                            m >= 236.0 && sdc <= 6.0 && dark >= 1
                        } else { false }
                    };
                    if cov_ok {
                        s.sub.push(2);
                        for k in 0..16 {
                            let mut c = js_round((255.0 - vv[k] as f64) / 36.0);
                            if c > 7 { c = 7; }
                            s.cov.push(c as u8);
                            let r = clamp_i(255 - c * 36);
                            let (px, py) = (ox + (k & 3), oy + (k >> 2)); if px < w && py < h { rec[py * w + px] = r as u8; is_base[py * w + px] = 1; }
                            let e = r as f64 - vv[k] as f64; sse += e * e;
                        }
                        continue;
                    }
                    // VQ + IBC4
                    let __t_nn = std::time::Instant::now();
                    let (a, b) = nn8_pair(cb, ke, &vv);
                    let qcl = if lq && has_line(ox, oy, 4) { 0 } else { qcls_of(&vv, lo, hi) };
                    let ql = qls[qcl];
                    let ctq = &ct[qcl];
                    let mut vbase = [0u8; 16];
                    for k in 0..16 { vbase[k] = if k < 8 { cb[a * 8 + k] } else { cb[b * 8 + (k - 8)] }; }
                    let vq_cost = resid_cost(&vv, &vbase, ctq, u64::MAX) + 16 * CSC;
                    t_nn += __t_nn.elapsed().as_nanos();
                    let mut ibc_cost = u64::MAX; let (mut idx_, mut idy_) = (0i32, 0i32);
                    let __t_i4 = std::time::Instant::now();
                    if vq_cost > 40 * CSC {
                        // УМНЫЙ поиск: предикторы соседей + локальное окно + refine (порядок = зеркало JS)
                        let gx4 = ox / 4; let gy4 = oy / 4;
                        let mut tried = [0u64; 32];
                        macro_rules! eval_c { ($dx:expr,$dy:expr) => {{
                            let dx = $dx; let dy = $dy;
                            let valid = !(dy < 0 || (dy == 0 && dx <= 0) || dx < -31 || dx > 31 || dy > 31);
                            let ti = if valid { (dy * 63 + dx + 31) as usize } else { 0 };
                            if valid && (!fast || tried[ti >> 6] & (1u64 << (ti & 63)) == 0) {
                                if fast { tried[ti >> 6] |= 1u64 << (ti & 63); }
                                let sx0 = ox as i32 - dx; let sy0 = oy as i32 - dy;
                                if sx0 >= 0 && sy0 >= 0 && sx0 as usize + 4 <= w && sy0 as usize + 4 <= h {
                                    let limit = ibc_cost.saturating_sub(10 * CSC);
                                    let lim32 = if limit > u32::MAX as u64 { u32::MAX } else { limit as u32 };
                                    let si0 = sy0 as usize * w + sx0 as usize;
                                    let mut ok;
                                    let mut cost = 0u32;
                                    #[cfg(target_arch = "x86_64")]
                                    {
                                        if simd::tier() >= 1 {
                                            let mut dif = [0u16; 16];
                                            ok = unsafe { simd::gather4_avx2(&vv, &rec, &is_base, si0, w, &mut dif) };
                                            if ok {
                                                for k in 0..16 { cost += ctq[dif[k] as usize]; }
                                                if cost >= lim32 { ok = false; }
                                            }
                                        } else {
                                            ok = true; let mut k4 = 0usize;
                                            'ocs: for y in 0..4usize { for x in 0..4usize {
                                                let si = si0 + y * w + x;
                                                if is_base[si] == 0 { ok = false; break 'ocs; }
                                                cost += ctq[(vv[k4] as i32 - rec[si] as i32 + 255) as usize];
                                                if cost >= lim32 { ok = false; break 'ocs; }
                                                k4 += 1;
                                            }}
                                        }
                                    }
                                    #[cfg(not(target_arch = "x86_64"))]
                                    {
                                        ok = true; let mut k4 = 0usize;
                                        'ocs: for y in 0..4usize { for x in 0..4usize {
                                            let si = si0 + y * w + x;
                                            if is_base[si] == 0 { ok = false; break 'ocs; }
                                            cost += ctq[(vv[k4] as i32 - rec[si] as i32 + 255) as usize];
                                            if cost >= limit { ok = false; break 'ocs; }
                                            k4 += 1;
                                        }}
                                    }
                                    if ok { let cost = cost as u64 + 10 * CSC; if cost < ibc_cost { ibc_cost = cost; idx_ = dx; idy_ = dy; } }
                                }
                            }
                        }} }
                        if fast {
                        // 1) предикторы: слева, сверху, сверху-слева, сверху-справа
                        if gx4 > 0 && gy4 < bh4 { let (px,py) = offs4[gy4*bw4+gx4-1]; if px!=0||py!=0 { eval_c!(px,py); } }
                        if gy4 > 0 && gx4 < bw4 { let (px,py) = offs4[(gy4-1)*bw4+gx4]; if px!=0||py!=0 { eval_c!(px,py); } }
                        if gx4 > 0 && gy4 > 0 { let (px,py) = offs4[(gy4-1)*bw4+gx4-1]; if px!=0||py!=0 { eval_c!(px,py); } }
                        if gy4 > 0 && gx4+1 < bw4 { let (px,py) = offs4[(gy4-1)*bw4+gx4+1]; if px!=0||py!=0 { eval_c!(px,py); } }
                        // early-accept: предиктор ~идеален → окно/far-grid/refine пропускаем
                        if ibc_cost > 14 * CSC {
                        // 2) локальное окно dy 0..4, dx -4..4
                        for dy in 0..=4i32 { let dxs = if dy>0 {-4} else {1}; for dx in dxs..=4i32 { eval_c!(dx,dy); } }
                        // far-grid зонды дальнего поля (+refine подберёт окрестность)
                        for dy in [8i32,12,16,20,24,28] { for dx in [-24i32,-16,-8,0,8,16,24] { eval_c!(dx,dy); } }
                        for dx in [8i32,12,16,20,24,28] { eval_c!(dx,0); }
                        // 3) refine ±2 вокруг лучшего, два раунда
                        for _round in 0..2 { let (bx2,by2)=(idx_,idy_); if ibc_cost != u64::MAX {
                            for dy in -2i32..=2 { for dx in -2i32..=2 { if dx==0&&dy==0 {continue;} eval_c!(bx2+dx,by2+dy); } } } }
                        }
                        } else {
                        // полный перебор — дефолт
                        for dy in 0..=31i32 { let dxs = if dy>0 {-31} else {1}; for dx in dxs..=31i32 { eval_c!(dx,dy); } }
                        }
                    }
                    t_ibc4 += __t_i4.elapsed().as_nanos();
                    // ИНТРА-кандидат (V/H/DC/diag от pass1-соседей) — зеркало JS opts.intra (теперь стандарт)
                    let __t_in = std::time::Instant::now();
                    let mut intra_cost = u64::MAX; let mut intra_im = 0u8; let mut intra_base = [0u8; 16];
                    {
                        let mut tt = [0u8; 4]; let mut ll = [0u8; 4];
                        let mut ok_t = oy > 0 && oy - 1 < h; let mut ok_l = ox > 0 && ox - 1 < w; // JS-зеркало: выход за буфер = undefined = false
                        if ok_t { for x in 0..4 { let xx = ox + x; if xx >= w || is_base[(oy - 1) * w + xx] == 0 { ok_t = false; break; } tt[x] = rec[(oy - 1) * w + xx]; } }
                        if ok_l { for y in 0..4 { let yy = oy + y; if yy >= h || is_base[yy * w + ox - 1] == 0 { ok_l = false; break; } ll[y] = rec[yy * w + ox - 1]; } }
                        let mut try_cand = |im: u8, bb: &[u8; 16], best: &mut u64, bim: &mut u8, bbase: &mut [u8; 16]| {
                            let limit = best.saturating_sub(5 * CSC);
                            let cost = resid_cost(&vv[..], &bb[..], ctq, limit).saturating_add(5 * CSC);
                            if cost < *best { *best = cost; *bim = im; *bbase = *bb; }
                        };
                        if ok_t { let mut b = [0u8; 16]; for k in 0..16 { b[k] = tt[k & 3]; } try_cand(0, &b, &mut intra_cost, &mut intra_im, &mut intra_base); }
                        if ok_l { let mut b = [0u8; 16]; for k in 0..16 { b[k] = ll[k >> 2]; } try_cand(1, &b, &mut intra_cost, &mut intra_im, &mut intra_base); }
                        if ok_t && ok_l { let mut sdc = 0i32; for i in 0..4 { sdc += tt[i] as i32 + ll[i] as i32; } let dc = js_round(sdc as f64 / 8.0); let mut b = [0u8; 16]; for k in 0..16 { b[k] = dc as u8; } try_cand(2, &b, &mut intra_cost, &mut intra_im, &mut intra_base); }
                        if ok_t { let mut b = [0u8; 16]; for k in 0..16 { b[k] = tt[((k & 3) + (k >> 2)).min(3)]; } try_cand(3, &b, &mut intra_cost, &mut intra_im, &mut intra_base); }
                    }
                    t_intra += __t_in.elapsed().as_nanos();
                    if intra_cost.saturating_mul(10) < vq_cost.min(ibc_cost).saturating_mul(9) {
                        s.sub.push(5); s.qcls.push(qcl as u8); s.imode.push(intra_im);
                        for k in 0..16 {
                            let q = js_round((vv[k] as f64 - intra_base[k] as f64) / ql as f64);
                            zig_push(&mut s.rz, q, qcl);
                            let r = clamp_i(intra_base[k] as i32 + q * ql);
                            let (px, py) = (ox + (k & 3), oy + (k >> 2)); if px < w && py < h { rec[py * w + px] = r as u8; }
                            let e = r as f64 - vv[k] as f64; sse += e * e;
                        }
                    } else if ibc_cost.saturating_mul(10) < vq_cost.saturating_mul(9) {
                        s.sub.push(3); s.qcls.push(qcl as u8);
                        s.ibcdx.push((idx_ & 0xff) as u8); s.ibcdy.push((idy_ & 0xff) as u8);
                        { let g4x = ox / 4; let g4y = oy / 4; if g4y < bh4 && g4x < bw4 { offs4[g4y*bw4+g4x] = (idx_, idy_); } }
                        let (sx0, sy0) = ((ox as i32 - idx_) as usize, (oy as i32 - idy_) as usize);
                        for k in 0..16 {
                            let bv = rec[(sy0 + (k >> 2)) * w + sx0 + (k & 3)];
                            let q = js_round((vv[k] as f64 - bv as f64) / ql as f64);
                            zig_push(&mut s.rz, q, qcl);
                            let r = clamp_i(bv as i32 + q * ql);
                            let (px, py) = (ox + (k & 3), oy + (k >> 2)); if px < w && py < h { rec[py * w + px] = r as u8; }
                            let e = r as f64 - vv[k] as f64; sse += e * e;
                        }
                    } else {
                        s.sub.push(4); s.qcls.push(qcl as u8);
                        s.it.push(a as u8); s.ib.push(b as u8);
                        for k in 0..16 {
                            let q = js_round((vv[k] as f64 - vbase[k] as f64) / ql as f64);
                            zig_push(&mut s.rz, q, qcl);
                            let r = clamp_i(vbase[k] as i32 + q * ql);
                            let (px, py) = (ox + (k & 3), oy + (k >> 2)); if px < w && py < h { rec[py * w + px] = r as u8; is_base[py * w + px] = 1; }
                            let e = r as f64 - vv[k] as f64; sse += e * e;
                        }
                    }
                }}
            }
        }
        s.m8.push(mcode);
    }}
    // NB: для pl8 (mcode 5) push сделан внутри ветки, а внешний пропущен через continue.
    if prof {
        let tot = t_start.elapsed().as_nanos();
        let ms = |n: u128| n as f64 / 1e6;
        eprintln!("PROF {}x{}: total {:.0}ms | det {:.0} dct {:.0} ibc8 {:.0} pl8 {:.0} nn {:.0} ibc4 {:.0} intra {:.0} other {:.0}",
            w, h, ms(tot), ms(t_det), ms(t_dct), ms(t_ibc8), ms(t_pl8), ms(t_nn), ms(t_ibc4), ms(t_intra),
            ms(tot.saturating_sub(t_det+t_dct+t_ibc8+t_pl8+t_nn+t_ibc4+t_intra)));
    }
    let (dm, dv, de) = zsplit_dmg(&s.dmg);
    let mut parts: Vec<Vec<u8>> = vec![
        pack3(&s.m8), s.sm, s.pc, s.pa, s.pb,
        pack_bits(&s.dsg), dm, dv, de,
        pack3(&s.sub), s.s4, s.p4c, s.p4a, s.p4b,
        pack3(&s.cov), s.ibcdx, s.ibcdy,
        s.it, s.ib, pack2(&s.imode),
    ];
    for c in 0..4 { let (mk, nz, esc) = zsplit_pack(&s.rz[c]); parts.push(mk); parts.push(nz); parts.push(esc); }
    parts.push(pack2(&s.qcls));
    (parts, sse, rec)
}

// ================= ХРОМА (порт _v16rustrun.js: quadPlane/coarseLuma/upGuided/cgain/chromaPatches) =================
// Числовая семантика JS: все промежуточные вычисления f64, планы хранятся f32/u8, Math.round=js_round.
// exp/hypot могут отличаться на 1 ULP от V8 — хрома-контракт по построению приближённый (GPU-шейдер декодера сам неточен).
const CS: usize = 8;

fn chroma_planes(data: &[u8], ch: usize, w: usize, h: usize, cs: usize) -> (Vec<u8>, Vec<u8>, usize, usize) {
    let sw = (w + cs - 1) / cs; let sh = (h + cs - 1) / cs;
    let mut a = vec![128u8; sw * sh]; let mut b = vec![128u8; sw * sh];
    if ch < 3 { return (a, b, sw, sh); }
    for gy in 0..sh { for gx in 0..sw {
        let (mut tr, mut tg, mut tb) = (0u32, 0u32, 0u32); let mut c = 0u32;
        if gx * cs + cs <= w && gy * cs + cs <= h {
            for dy in 0..cs {
                let base = ((gy * cs + dy) * w + gx * cs) * ch;
                let rowc = &data[base..base + cs * ch];
                for px in rowc.chunks_exact(ch) { tr += px[0] as u32; tg += px[1] as u32; tb += px[2] as u32; }
            }
            c = (cs * cs) as u32;
        } else {
        for dy in 0..cs { for dx in 0..cs {
            let yy = gy * cs + dy; let xx = gx * cs + dx;
            if yy < h && xx < w {
                let o = (yy * w + xx) * ch;
                tr += data[o] as u32; tg += data[o + 1] as u32; tb += data[o + 2] as u32;
                c += 1;
            }
        }}
        }
        let sb = 128.0 * c as f64 - 0.168736 * tr as f64 - 0.331264 * tg as f64 + 0.5 * tb as f64;
        let sr = 128.0 * c as f64 + 0.5 * tr as f64 - 0.418688 * tg as f64 - 0.081312 * tb as f64;
        a[gy * sw + gx] = clamp_i(js_round(sb / c as f64)) as u8;
        b[gy * sw + gx] = clamp_i(js_round(sr / c as f64)) as u8;
    }}
    (a, b, sw, sh)
}

struct QuadOut { parts: Vec<Vec<u8>>, rec: Vec<u8> } // parts: [fb, pc, pa, pb, resid] (quad) | [mask,vals,esc] (dct)
// DCT-8 матрица для хромы (та же ортонормированная, что и у люма-DCT)
fn c8_chroma() -> &'static [[f64; 8]; 8] {
    static C: std::sync::OnceLock<[[f64; 8]; 8]> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        let mut c = [[0.0f64; 8]; 8];
        for u in 0..8 { for x in 0..8 {
            c[u][x] = ((2.0 * x as f64 + 1.0) * u as f64 * std::f64::consts::PI / 16.0).cos()
                * if u == 0 { (1.0f64 / 8.0).sqrt() } else { (2.0f64 / 8.0).sqrt() };
        }}
        c
    })
}
// зигзаг-порядок 8×8 (низкие частоты первыми) — группирует нули в хвост блока для EOB
static ZIGZAG: [usize; 64] = [
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63];
// DCT-хрома плана: 8×8 DCT coarse-плана → квант qdc → зигзаг + EOB (хранить только коэфф до последнего
// ненулевого; хвост нулей не кодируется — гладкая хрома = 95% нулей). rec = IDCT (переупаковка lossless).
// parts: [eob (u8/блок), coeff mask/vals/esc (zig u16, только до EOB)].
fn dct_plane(p: &[u8], sw: usize, sh: usize, c8: &[[f64; 8]; 8], qdc: f64) -> QuadOut {
    let bw = (sw + 7) / 8; let bh = (sh + 7) / 8;
    let mut rec = vec![0u8; sw * sh];
    let mut eobs: Vec<u8> = Vec::with_capacity(bw * bh);
    let mut coeff: Vec<u16> = Vec::with_capacity(bw * bh * 8);
    // env + перцептивная квант-матрица считаются ОДИН раз на плоскость (не на блок): раньше envf (getenv+alloc)
    // и qeff-пересчёт сидели в цикле блоков → ~800k getenv + 51M float/главу впустую.
    let dz = envf("MVQ_CHDCT_DZ", 1.0) as i32; // deadzone: |AC-квант|<=dz → 0 (убрать мелкий AC-шум, DC не трогать)
    // перцептивная квант-матрица: qeff = qdc*(1+slope*(u+v)) — низкие частоты точнее, высокие грубее
    // (принцип JPEG/VP8). slope=0 → плоский квант. rec использует то же qeff (обратимо).
    let slope = envf("MVQ_CHQSLOPE", 0.22);
    let mut qeff_t = [0.0f64; 64];
    for k in 0..64 { let freq = (k & 7) + (k >> 3); qeff_t[k] = qdc * (1.0 + slope * freq as f64); }
    for by in 0..bh { for bx in 0..bw {
        let mut bl = [0.0f64; 64];
        for y in 0..8 { for x in 0..8 {
            let sx = (bx * 8 + x).min(sw - 1); let sy = (by * 8 + y).min(sh - 1);
            bl[y * 8 + x] = p[sy * sw + sx] as f64;
        }}
        let mut co = [0.0f64; 64]; dct8(&bl, &mut co, c8);
        let mut qs = [0i32; 64]; let mut qd = [0.0f64; 64];
        for k in 0..64 {
            let qeff = qeff_t[k];
            let mut q = js_round(co[k] / qeff); if k != 0 && q.abs() <= dz { q = 0; } qs[k] = q; qd[k] = q as f64 * qeff;
        }
        let mut rc = [0.0f64; 64]; idct8(&qd, &mut rc, c8);
        for y in 0..8 { for x in 0..8 { let sx = bx * 8 + x; let sy = by * 8 + y;
            if sx < sw && sy < sh { rec[sy * sw + sx] = clamp_i(js_round(rc[y * 8 + x])) as u8; } }}
        // зигзаг → EOB (индекс последнего ненулевого +1)
        let mut z = [0i32; 64];
        for i in 0..64 { z[i] = qs[ZIGZAG[i]]; }
        let mut eob = 0usize;
        for i in (0..64).rev() { if z[i] != 0 { eob = i + 1; break; } }
        eobs.push(eob as u8);
        for i in 0..eob { coeff.push(z[i] as i16 as u16); }
    }}
    let (m, v, e) = zsplit_dmg(&coeff);
    QuadOut { parts: vec![eobs, m, v, e], rec }
}
fn quad_fit(p: &[u8], sw: usize, sh: usize, x0: usize, y0: usize, n: usize) -> Option<(i32, i32, i32, f64)> {
    let cxy = (n as f64 - 1.0) / 2.0;
    let mut sum = 0.0f64; let mut cnt = 0u32;
    for y in 0..n { for x in 0..n { let xx = x0 + x; let yy = y0 + y; if xx < sw && yy < sh { sum += p[yy * sw + xx] as f64; cnt += 1; } } }
    if cnt == 0 { return None; }
    let c = js_round(sum / cnt as f64);
    let (mut sxx, mut sxv, mut syy, mut syv) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..n { for x in 0..n { let xx = x0 + x; let yy = y0 + y; if xx < sw && yy < sh {
        let v = p[yy * sw + xx] as f64; let dx = x as f64 - cxy; let dy = y as f64 - cxy;
        sxx += dx * dx; sxv += dx * (v - c as f64); syy += dy * dy; syv += dy * (v - c as f64);
    } } }
    let ga = if sxx != 0.0 { js_round(sxv / sxx) } else { 0 };
    let gb = if syy != 0.0 { js_round(syv / syy) } else { 0 };
    let mut me = 0.0f64;
    for y in 0..n { for x in 0..n { let xx = x0 + x; let yy = y0 + y; if xx < sw && yy < sh {
        let r = clamp_f(c as f64 + ga as f64 * (x as f64 - cxy) + gb as f64 * (y as f64 - cxy));
        let e = (r - p[yy * sw + xx] as f64).abs(); if e > me { me = e; }
    } } }
    Some((c, ga, gb, me))
}
#[allow(clippy::too_many_arguments)]
fn quad_chq()->i32{static Q:std::sync::OnceLock<i32>=std::sync::OnceLock::new();*Q.get_or_init(||std::env::var("MVQ_CHQ").ok().and_then(|v|v.parse().ok()).unwrap_or(8))}
fn quad_enc(p: &[u8], sw: usize, sh: usize, x0: usize, y0: usize, n: usize,
            flags: &mut Vec<u8>, pc: &mut Vec<u8>, pa: &mut Vec<u8>, pb: &mut Vec<u8>, resid: &mut Vec<u8>, rec: &mut [u8]) {
    let CHQ = quad_chq();
    let f = match quad_fit(p, sw, sh, x0, y0, n) { Some(f) => f, None => return };
    let (c, ga, gb, me) = f;
    if me <= 4.0 || n == 4 {
        flags.push(0); pc.push((c & 255) as u8); pa.push((ga & 255) as u8); pb.push((gb & 255) as u8);
        let cxy = (n as f64 - 1.0) / 2.0;
        for y in 0..n { for x in 0..n { let xx = x0 + x; let yy = y0 + y; if xx < sw && yy < sh {
            let r = clamp_f(c as f64 + ga as f64 * (x as f64 - cxy) + gb as f64 * (y as f64 - cxy));
            let q = js_round((p[yy * sw + xx] as f64 - r) / CHQ as f64);
            let z = if q >= 0 { 2 * q } else { -2 * q - 1 };
            resid.push((z & 255) as u8);
            // JS: rec[i]=clamp(r+q*8) — запись float в Uint8Array усечением к нулю
            rec[yy * sw + xx] = clamp_f(r + (q * CHQ) as f64) as u8;
        } } }
        return;
    }
    flags.push(1); let h2 = n >> 1;
    quad_enc(p, sw, sh, x0, y0, h2, flags, pc, pa, pb, resid, rec);
    quad_enc(p, sw, sh, x0 + h2, y0, h2, flags, pc, pa, pb, resid, rec);
    quad_enc(p, sw, sh, x0, y0 + h2, h2, flags, pc, pa, pb, resid, rec);
    quad_enc(p, sw, sh, x0 + h2, y0 + h2, h2, flags, pc, pa, pb, resid, rec);
}
fn quad_plane(p: &[u8], sw: usize, sh: usize) -> QuadOut {
    let mut flags = Vec::new(); let mut pc = Vec::new(); let mut pa = Vec::new(); let mut pb = Vec::new(); let mut resid = Vec::new();
    let mut rec = vec![0u8; sw * sh];
    let mut y0 = 0usize;
    while y0 < sh { let mut x0 = 0usize; while x0 < sw { quad_enc(p, sw, sh, x0, y0, 32, &mut flags, &mut pc, &mut pa, &mut pb, &mut resid, &mut rec); x0 += 32; } y0 += 32; }
    QuadOut { parts: vec![pack_bits(&flags), pc, pa, pb, resid], rec }
}

fn coarse_luma(y_plane: &[u8], w: usize, h: usize, cs: usize) -> Vec<f32> {
    let sw = (w + cs - 1) / cs; let sh = (h + cs - 1) / cs;
    let mut o = vec![0f32; sw * sh];
    for gy in 0..sh { for gx in 0..sw {
        let mut s = 0.0f64; let mut c = 0u32;
        for dy in 0..cs { for dx in 0..cs { let yy = gy * cs + dy; let xx = gx * cs + dx; if yy < h && xx < w { s += y_plane[yy * w + xx] as f64; c += 1; } } }
        o[gy * sw + gx] = (s / c as f64) as f32;
    }}
    o
}

// Кубический B-spline базис-веса для t∈[0,1] по 4 узлам (P-1,P0,P1,P2). Все веса ≥0, сумма=1 →
// НЕТ overshoot/звона на резких хрома-краях (в отличие от Catmull-Rom). C2-гладкий, сглаживающий
// (не интерполирует узлы точно — лёгкое размытие, что против блочности даже в плюс).
#[inline] fn envf(name: &str, def: f64) -> f64 { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(def) }
#[inline] fn cr_weights(t: f32) -> [f32; 4] {
    let t2 = t * t; let t3 = t2 * t; let it = 1.0 - t;
    [it * it * it / 6.0,
     (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
     (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
     t3 / 6.0]
}
// mvq: SSE-версия внутреннего 4×4-tap ядра up_guided2. 4 i-tap'а = 4 lane f32.
// НЕ bit-exact к скаляру (vector hsum меняет порядок суммирования на последний ULP) —
// хрома приближённая (residual re-correct), дрейф sub-quant. Fallback — скаляр (см. вызов).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum128(v: std::arch::x86_64::__m128) -> f32 {
    use std::arch::x86_64::*;
    let sh = _mm_movehl_ps(v, v);
    let s2 = _mm_add_ps(v, sh);
    let sh2 = _mm_shuffle_ps(s2, s2, 0x55);
    _mm_cvtss_f32(_mm_add_ss(s2, sh2))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn guided_pixel_sse(cl: &[f32], pa: &[u8], pb: &[u8], cxc: &[usize; 4], ryc: &[usize; 4], sw: usize, yl: f32, inv_s2: f32, wx: &[f32; 4], wy: &[f32; 4]) -> (f32, f32, f32) {
    use std::arch::x86_64::*;
    let ylv = _mm_set1_ps(yl);
    let invv = _mm_set1_ps(inv_s2);
    let c0625 = _mm_set1_ps(0.0625);
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let wxv = _mm_loadu_ps(wx.as_ptr());
    let contig = cxc[1] == cxc[0] + 1 && cxc[2] == cxc[0] + 2 && cxc[3] == cxc[0] + 3;
    let (mut sa, mut sb, mut sws) = (zero, zero, zero);
    for j in 0..4 {
        let row = ryc[j] * sw;
        let (clv, pav, pbv);
        if contig {
            let base = row + cxc[0];
            clv = _mm_loadu_ps(cl.as_ptr().add(base));
            pav = _mm_set_ps(pa[base + 3] as f32, pa[base + 2] as f32, pa[base + 1] as f32, pa[base] as f32);
            pbv = _mm_set_ps(pb[base + 3] as f32, pb[base + 2] as f32, pb[base + 1] as f32, pb[base] as f32);
        } else {
            let (i0, i1, i2, i3) = (row + cxc[0], row + cxc[1], row + cxc[2], row + cxc[3]);
            clv = _mm_set_ps(cl[i3], cl[i2], cl[i1], cl[i0]);
            pav = _mm_set_ps(pa[i3] as f32, pa[i2] as f32, pa[i1] as f32, pa[i0] as f32);
            pbv = _mm_set_ps(pb[i3] as f32, pb[i2] as f32, pb[i1] as f32, pb[i0] as f32);
        }
        let dl = _mm_sub_ps(ylv, clv);
        let xq = _mm_mul_ps(_mm_mul_ps(dl, dl), invv);
        let t = _mm_max_ps(_mm_sub_ps(one, _mm_mul_ps(xq, c0625)), zero);
        let t2 = _mm_mul_ps(t, t);
        let t4 = _mm_mul_ps(t2, t2);
        let t8 = _mm_mul_ps(t4, t4);
        let t16 = _mm_mul_ps(t8, t8);
        let wgt = _mm_mul_ps(_mm_mul_ps(wxv, _mm_set1_ps(wy[j])), t16);
        sa = _mm_add_ps(sa, _mm_mul_ps(pav, wgt));
        sb = _mm_add_ps(sb, _mm_mul_ps(pbv, wgt));
        sws = _mm_add_ps(sws, wgt);
    }
    (hsum128(sa), hsum128(sb), hsum128(sws))
}
fn up_guided2(pa: &[u8], pb: &[u8], cl: &[f32], sw: usize, sh: usize, w: usize, h: usize, yrec: &[u8], sigma: f64, cs: usize) -> (Vec<f32>, Vec<f32>) {
    // БИКУБИЧЕСКИЙ guided-апсемпл (2026-07-06): 4×4 узла Catmull-Rom × люма-модуляция pw(ΔY).
    // Билинейный (2×2) фасетил градиент на границах 8px-ячеек; CR C1-гладкий. σ↑ ослабляет люма-привязку
    // (убирает перенос люма-текстуры волос/ткани в хрому). f32, хрома-контракт приближённый.
    let mut oa = vec![0f32; w * h]; let mut ob = vec![0f32; w * h]; let s2 = (2.0 * sigma * sigma) as f32;
    let _ = s2;
    // адаптивный σ по хрома-разбросу узлов (2026-07-06): гладь/градиент одного цвета → большой σ (B-spline
    // гладко, правило B); граница разных цветов → малый σ (жёсткая привязка к своей стороне, без серого
    // ореола, правило A). σ_эфф = lerp(σ_gran, σ_glad) по spread хромы 16 узлов. env-ручки для тюнинга.
    let sig_gl = envf("MVQ_SIGMA", 60.0) as f32;   // гладь
    let sig_gr = envf("MVQ_SIGMA_EDGE", 10.0) as f32; // граница
    let spr_t = envf("MVQ_CHSPREAD", 18.0) as f32;    // порог хрома-разброса = граница
    let xt: Vec<(isize, [f32; 4])> = (0..w).map(|x| {
        let fx = (x as f64 + 0.5) / cs as f64 - 0.5;
        let ix = fx.floor();
        (ix as isize, cr_weights((fx - ix) as f32))
    }).collect();
    let clampi = |v: isize, hi: usize| -> usize { if v < 0 { 0 } else if v as usize >= hi { hi - 1 } else { v as usize } };
    // Предпосчёт узловых индексов + адаптивного inv_s2 по ВСЕЙ coarse-сетке (2026-07-07): step-детектор,
    // cxc/ryc и inv_s2 зависят ТОЛЬКО от ячейки (ix,iy) — грид считается ОДИН раз (coarse-размер), в
    // пиксельном цикле O(1) lookup вместо пересчёта. Бит-точно (те же значения), step ушёл из горячего пути.
    let gcw = sw + 2; let gch = sh + 2; // ix∈[-1..sw] → индекс [0..sw+1]
    let cxc_by: Vec<[usize; 4]> = (0..gcw).map(|c| { let ix = c as isize - 1;
        [clampi(ix - 1, sw), clampi(ix, sw), clampi(ix + 1, sw), clampi(ix + 2, sw)] }).collect();
    let ryc_by: Vec<[usize; 4]> = (0..gch).map(|c| { let iy = c as isize - 1;
        [clampi(iy - 1, sh), clampi(iy, sh), clampi(iy + 1, sh), clampi(iy + 2, sh)] }).collect();
    let mut inv_grid = vec![0f32; gcw * gch];
    for iyc in 0..gch { let ryc = ryc_by[iyc];
        for ixc in 0..gcw { let cxc = cxc_by[ixc];
            let mut step = 0f32;
            for j in 0..4 { let row = ryc[j] * sw;
                for i in 0..3 { let a = pa[row+cxc[i]] as f32 - pa[row+cxc[i+1]] as f32;
                    let b = pb[row+cxc[i]] as f32 - pb[row+cxc[i+1]] as f32;
                    let s = a.abs().max(b.abs()); if s > step { step = s; } } }
            for i in 0..4 { for j in 0..3 { let a = pa[ryc[j]*sw+cxc[i]] as f32 - pa[ryc[j+1]*sw+cxc[i]] as f32;
                    let b = pb[ryc[j]*sw+cxc[i]] as f32 - pb[ryc[j+1]*sw+cxc[i]] as f32;
                    let s = a.abs().max(b.abs()); if s > step { step = s; } } }
            let f = (step / spr_t).min(1.0); // 0=плавно(гладь/градиент), 1=резкий скачок(граница)
            let sig = sig_gl + (sig_gr - sig_gl) * f;
            inv_grid[iyc * gcw + ixc] = 1.0f32 / (2.0 * sig * sig);
        } }
    #[cfg(target_arch = "x86_64")]
    let use_guided_simd = std::is_x86_feature_detected!("avx2") && std::env::var_os("MVQ_NOCHSIMD").is_none();
    #[cfg(not(target_arch = "x86_64"))]
    let use_guided_simd = false;
    for y in 0..h {
        let fy = (y as f64 + 0.5) / cs as f64 - 0.5;
        let iy = fy.floor() as isize;
        let wy = cr_weights((fy as f32) - iy as f32);
        let iyc = (iy + 1) as usize;
        let ryc = ryc_by[iyc];
        for x in 0..w {
            let (ix, wx) = xt[x];
            let ixc = (ix + 1) as usize;
            let cxc = cxc_by[ixc];
            let inv_s2 = inv_grid[iyc * gcw + ixc];
            let yl = yrec[y * w + x] as f32;
            let (sa, sb, ws) = if use_guided_simd {
                unsafe { guided_pixel_sse(cl, pa, pb, &cxc, &ryc, sw, yl, inv_s2, &wx, &wy) }
            } else {
                let (mut sa, mut sb, mut ws) = (0.0f32, 0.0f32, 0.0f32);
                for j in 0..4 {
                    let row = ryc[j] * sw;
                    for i in 0..4 {
                        let idx = row + cxc[i];
                        let dl = yl - cl[idx];
                        let xq = dl * dl * inv_s2;
                        let t = (1.0 - xq * 0.0625).max(0.0);
                        let t2 = t * t; let t4 = t2 * t2; let t8 = t4 * t4;
                        let wgt = wx[i] * wy[j] * (t8 * t8);
                        sa += pa[idx] as f32 * wgt;
                        sb += pb[idx] as f32 * wgt;
                        ws += wgt;
                    }
                }
                (sa, sb, ws)
            };
            if ws.abs() > 1e-6 {
                let inv = 1.0 / ws;
                oa[y * w + x] = sa * inv;
                ob[y * w + x] = sb * inv;
            } else {
                let idx = clampi(iy, sh) * sw + clampi(ix, sw);
                oa[y * w + x] = pa[idx] as f32; ob[y * w + x] = pb[idx] as f32;
            }
        }
    }
    (oa, ob)
}
#[allow(dead_code)]
fn up_guided(p: &[u8], cl: &[f32], sw: usize, sh: usize, w: usize, h: usize, yrec: &[u8], sigma: f64) -> Vec<f32> {
    let mut o = vec![0f32; w * h]; let s2 = 2.0 * sigma * sigma;
    // LUT exp(-(dl^2)/s2) по |dl| с шагом 0.25 + lerp (|dl|<=255); дрейф от точного exp < 1e-4 — хрома-контракт приближённый
    let mut lut = [0f64; 1026];
    for i in 0..1026 { let d = i as f64 * 0.25; lut[i] = (-(d * d) / s2).exp(); }
    let expw = |dl: f64| -> f64 { let a = dl.abs() * 4.0; let i = a as usize; if i >= 1024 { 0.0 } else { let f = a - i as f64; lut[i] * (1.0 - f) + lut[i + 1] * f } };
    // x-таблицы один раз на кадр (значения идентичны попиксельному пересчёту)
    let mut xt: Vec<(usize, usize, f64)> = Vec::with_capacity(w);
    for x in 0..w {
        let fx = (x as f64 + 0.5) / CS as f64 - 0.5;
        let x0 = fx.floor().min(sw as f64 - 1.0).max(0.0) as usize;
        let x1 = (x0 + 1).min(sw - 1);
        let ax = (fx - x0 as f64).max(0.0).min(1.0);
        xt.push((x0, x1, ax));
    }
    for y in 0..h {
        let fy = (y as f64 + 0.5) / CS as f64 - 0.5;
        let y0 = fy.floor().min(sh as f64 - 1.0).max(0.0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ay = (fy - y0 as f64).max(0.0).min(1.0);
        for x in 0..w {
            let (x0, x1, ax) = xt[x];
            let yl = yrec[y * w + x] as f64;
            let (t00, t10, t01, t11) = ((1.0 - ax) * (1.0 - ay), ax * (1.0 - ay), (1.0 - ax) * ay, ax * ay);
            let mut sum = 0.0f64; let mut wsum = 0.0f64;
            let mut dl = yl - cl[y0 * sw + x0] as f64; let mut wl = t00 * expw(dl); sum += p[y0 * sw + x0] as f64 * wl; wsum += wl;
            dl = yl - cl[y0 * sw + x1] as f64; wl = t10 * expw(dl); sum += p[y0 * sw + x1] as f64 * wl; wsum += wl;
            dl = yl - cl[y1 * sw + x0] as f64; wl = t01 * expw(dl); sum += p[y1 * sw + x0] as f64 * wl; wsum += wl;
            dl = yl - cl[y1 * sw + x1] as f64; wl = t11 * expw(dl); sum += p[y1 * sw + x1] as f64 * wl; wsum += wl;
            o[y * w + x] = if wsum > 1e-6 { (sum / wsum) as f32 } else { p[y0 * sw + x0] as f32 };
        }
    }
    o
}

pub static CH_PROF: [core::sync::atomic::AtomicU64; 5] = [
    core::sync::atomic::AtomicU64::new(0), // planes
    core::sync::atomic::AtomicU64::new(0), // quad
    core::sync::atomic::AtomicU64::new(0), // guided(+coarse)
    core::sync::atomic::AtomicU64::new(0), // cgain
    core::sync::atomic::AtomicU64::new(0), // patches
];
#[inline] fn chp(i: usize, t: std::time::Instant) { CH_PROF[i].fetch_add(t.elapsed().as_micros() as u64, core::sync::atomic::Ordering::Relaxed); }
pub struct ChromaOut { k: f32, parts: Vec<Vec<u8>>, planes: Option<(Vec<u8>, Vec<u8>)>, coarse: Option<(Vec<u8>, Vec<u8>, u32, u32)> }
fn encode_chroma(data: &[u8], ch: usize, w: usize, h: usize, luma_rec: &[u8], emit: bool) -> ChromaOut {
    let __t = std::time::Instant::now();
    let cs = (envf("MVQ_CS", 2.0) as usize).max(1);
    let (pla, plb, sw, sh) = chroma_planes(data, ch, w, h, cs);
    chp(0, __t);
    let __t = std::time::Instant::now();
    // MVQ_CHMODE=1 → DCT-хрома (гладкие базисы, нет изломов планарных листьев); иначе quad (планары)
    let chmode = envf("MVQ_CHMODE", 1.0) as i32;
    let (qa, qb) = if chmode == 2 {
        // RAW (неблочная хрома): coarse-план хранится НАПРЯМУЮ, без поблочного DCT/quad → нет квадратной
        // решётки от блок-кодирования. Гладкий coarse (box-downsample) + guided upsample = единая гладкая
        // поверхность. Вес = brotli(сырой coarse). Тест гипотезы «поблочность = источник квадратов хромы».
        (QuadOut { parts: vec![pla.clone()], rec: pla.clone() }, QuadOut { parts: vec![plb.clone()], rec: plb.clone() })
    } else if chmode == 1 {
        let qdc = envf("MVQ_CHDCT_Q", 3.0); // 3 (не 14): DCT-квант хромы низкочастотит цвет по 16px-блокам →
        // bleeding (цвет течёт через границу на соседний объект: розовый ореол у чёрных линий/прядей на белом).
        // qdc=3 сохраняет высокие частоты хромы → границы цвета резкие, ореол уходит (как S1, но без цены S1
        // — разрешение и время guided те же). Корпус: −21.1% под webp (было −24.1%), ~3 п.п. за чистый цвет.
        (dct_plane(&pla, sw, sh, c8_chroma(), qdc), dct_plane(&plb, sw, sh, c8_chroma(), qdc))
    } else {
        (quad_plane(&pla, sw, sh), quad_plane(&plb, sw, sh))
    };
    chp(1, __t);
    let mut parts = qa.parts; let qa_rec = qa.rec; let qb_rec = qb.rec;
    parts.extend(qb.parts);
    // STRUCTDUMP: coarse-recon хромы (квантованные ступеньки ДО guided/upsample/фильтров = чистая структура
    // кодирования) — для объективного структурного banding-детектора (не пиксельного). Только при emit.
    let coarse_dump = if emit { Some((qa_rec.clone(), qb_rec.clone(), sw as u32, sh as u32)) } else { None };
    if ch < 3 {
        parts.push(Vec::new()); parts.push(Vec::new()); // pmap, zz пустые
        return ChromaOut { k: 1.0, parts, planes: if emit { Some((vec![128u8; w * h], vec![128u8; w * h])) } else { None }, coarse: coarse_dump };
    }
    let n = w * h;
    // guided-апсемпл по рекон-люме (σ=14)
    let __t = std::time::Instant::now();
    let cl = coarse_luma(luma_rec, w, h, cs);
    let sigma: f64 = { static S: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
        *S.get_or_init(|| std::env::var("MVQ_SIGMA").ok().and_then(|v| v.parse().ok()).unwrap_or(14.0)) };
    let (mut up_b, mut up_r) = up_guided2(&qa_rec, &qb_rec, &cl, sw, sh, w, h, luma_rec, sigma, cs);
    chp(2, __t);
    let __t = std::time::Instant::now();
    // per-page sat-gain k (МНК-фит по каждому 2-му пикселю, гейт so>10)
    let mut num = 0.0f64; let mut den = 0.0f64;
    let mut pix = 0usize;
    while pix < n {
        let o = pix * ch;
        let r = data[o] as f64; let g = data[o + 1] as f64; let bb = data[o + 2] as f64;
        let cbo = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * bb;
        let cro = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * bb;
        let so2 = { let a = cbo - 128.0; let b = cro - 128.0; a * a + b * b };
        if so2 > 100.0 {
            let so = so2.sqrt();
            let sr = { let a = up_b[pix] as f64 - 128.0; let b = up_r[pix] as f64 - 128.0; (a * a + b * b).sqrt() };
            num += so * sr; den += sr * sr;
        }
        pix += 4; // было 2: k = МНК-фит, половина выборки сдвигает k на ~0.001 (дрейф-класс)
    }
    let k = if den > 0.0 { (num / den).max(1.0).min(1.25) } else { 1.0 };
    if k != 1.0 {
        for i in 0..n {
            up_b[i] = (128.0 + (up_b[i] as f64 - 128.0) * k) as f32;
            up_r[i] = (128.0 + (up_r[i] as f64 - 128.0) * k) as f32;
        }
    }
    chp(3, __t);
    let __t = std::time::Instant::now();
    // sat-aware патчи S=4 q6 поверх guided+gain (гейт err>12 или vivid-drop>8)
    let sw8 = (w + 7) / 8; let sh8 = (h + 7) / 8;
    let mut pmap = vec![0u8; sw8 * sh8]; let mut pd: Vec<i32> = Vec::new();
    const S: usize = 4; const QDP: f64 = 6.0;
    let mut cbc = [0.0f32; 64]; let mut crc = [0.0f32; 64]; let mut valid = [false; 64];
    for gy in 0..sh8 { for gx in 0..sw8 {
        let mut me = 0.0f32; let mut vivid = false; let mut drop_max = 0.0f32;
        let inner_blk = gx * 8 + 8 <= w && gy * 8 + 8 <= h;

        if inner_blk {
            for dy in 0..8 {
                let yy = gy * 8 + dy;
                let rowd = &data[(yy * w + gx * 8) * ch..(yy * w + gx * 8 + 8) * ch];
                let ub = &up_b[yy * w + gx * 8..yy * w + gx * 8 + 8];
                let ur = &up_r[yy * w + gx * 8..yy * w + gx * 8 + 8];
                for dx in 0..8 {
                    let kk = dy * 8 + dx;
                    valid[kk] = true;
                    let o = dx * ch;
                    let r = rowd[o] as f32; let g = rowd[o + 1] as f32; let bb = rowd[o + 2] as f32;
                    let cbv = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * bb;
                    let crv = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * bb;
                    cbc[kk] = cbv; crc[kk] = crv;
                    let e = (cbv - ub[dx]).abs().max((crv - ur[dx]).abs());
                    if e > me { me = e; }
                    let so2 = { let a = cbv - 128.0; let b = crv - 128.0; a * a + b * b };
                    if so2 > 1600.0 {
                        vivid = true;
                        let so = so2.sqrt();
                        let sr = { let a = ub[dx] - 128.0; let b = ur[dx] - 128.0; (a * a + b * b).sqrt() };
                        if so - sr > drop_max { drop_max = so - sr; }
                    }
                }
            }
        } else {
        for dy in 0..8 { for dx in 0..8 {
            let yy = gy * 8 + dy; let xx = gx * 8 + dx;
            let kk = dy * 8 + dx;
            if yy >= h || xx >= w { valid[kk] = false; continue; }
            valid[kk] = true;
            let o = (yy * w + xx) * ch;
            let r = data[o] as f32; let g = data[o + 1] as f32; let bb = data[o + 2] as f32;
            let cbv = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * bb;
            let crv = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * bb;
            cbc[kk] = cbv; crc[kk] = crv;
            let i = yy * w + xx;
            let e = (cbv - up_b[i]).abs().max((crv - up_r[i]).abs());
            if e > me { me = e; }
            let so2 = { let a = cbv - 128.0; let b = crv - 128.0; a * a + b * b };
            if so2 > 1600.0 {
                vivid = true;
                let so = so2.sqrt();
                let sr = { let a = up_b[i] - 128.0; let b = up_r[i] - 128.0; (a * a + b * b).sqrt() };
                if so - sr > drop_max { drop_max = so - sr; }
            }
        }}
        }
        if !(me > 12.0f32 || (vivid && drop_max > 8.0f32) ) { continue; }
        pmap[gy * sw8 + gx] = 1;
        for chn in 0..2 { for sy in 0..2 { for sx in 0..2 {
            let mut sum = 0.0f32; let mut c = 0u32;
            for dy in 0..S { for dx in 0..S {
                let yy = gy * 8 + sy * S + dy; let xx = gx * 8 + sx * S + dx;
                let kk = (sy * S + dy) * 8 + sx * S + dx;
                if !valid[kk] { continue; }
                let v = if chn == 0 { cbc[kk] } else { crc[kk] };
                let up = if chn == 0 { up_b[yy * w + xx] } else { up_r[yy * w + xx] };
                sum += v - up; c += 1;
            }}
            let q = js_round(if c > 0 { sum as f64 / c as f64 } else { 0.0 } / QDP);
            pd.push(q);
        }}}
    }}
    let zz: Vec<u8> = pd.iter().map(|&q| { let z = if q >= 0 { 2 * q } else { -2 * q - 1 }; z.min(255) as u8 }).collect();
    parts.push(pack_bits(&pmap));
    parts.push(zz);
    chp(4, __t);
    let planes = if emit {
        // применяем патчи на float-планы, затем u8 (js_round+clamp)
        let mut pi = 0usize;
        for gy in 0..sh8 { for gx in 0..sw8 {
            if pmap[gy * sw8 + gx] == 0 { continue; }
            for chn in 0..2 { for sy in 0..2 { for sx in 0..2 {
                let dqd = pd[pi] as f64 * QDP; pi += 1;
                for dy in 0..S { for dx in 0..S {
                    let yy = gy * 8 + sy * S + dy; let xx = gx * 8 + sx * S + dx;
                    if yy >= h || xx >= w { continue; }
                    let i = yy * w + xx;
                    if chn == 0 { up_b[i] = (up_b[i] as f64 + dqd) as f32; } else { up_r[i] = (up_r[i] as f64 + dqd) as f32; }
                }}
            }}}
        }}
        let mut cbp: Vec<u8> = up_b.iter().map(|&v| clamp_i(js_round(v as f64)) as u8).collect();
        let mut crp: Vec<u8> = up_r.iter().map(|&v| clamp_i(js_round(v as f64)) as u8).collect();
        let cht: i32 = { static T: std::sync::OnceLock<i32> = std::sync::OnceLock::new(); *T.get_or_init(|| std::env::var("MVQ_CHSMOOTH").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) };
        if cht > 0 {
            // сглаживание Cb/Cr в зонах гладкой хромы (гейт по локальному размаху 3x3 ≤ cht) — гасит 8px-фасетки, края цвета не трогает
            for (pl, _tag) in [(&mut cbp, 0u8), (&mut crp, 1u8)] {
                let src = pl.clone();
                for y in 1..h-1 { for x in 1..w-1 {
                    let i = y * w + x;
                    let (mut mn, mut mx) = (255i32, 0i32);
                    for dy in -1i32..=1 { for dx in -1i32..=1 { let v = src[(i as i32 + dy * w as i32 + dx) as usize] as i32; if v < mn { mn = v; } if v > mx { mx = v; } } }
                    if mx - mn <= cht {
                        let s = 4 * src[i] as i32 + src[i-1] as i32 + src[i+1] as i32 + src[i-w] as i32 + src[i+w] as i32;
                        pl[i] = js_round(s as f64 / 8.0) as u8;
                    }
                }}
            }
        }
        // хрома-деблок (loop-filter-аналог, 0 байт): сглаживание перепада ТОЛЬКО на границах coarse-ячеек
        // (каждые cs px) при малом скачке — гасит блочность S-сетки, реальные цвет-границы (большой скачок) не трогает.
        let chdb: i32 = { static T: std::sync::OnceLock<i32> = std::sync::OnceLock::new(); *T.get_or_init(|| std::env::var("MVQ_CHDEBLOCK").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) };
        if chdb > 0 {
            for pl in [&mut cbp, &mut crp] {
                let src = pl.clone();
                // вертикальные границы ячеек
                for y in 0..h { let mut x = cs; while x < w { let i = y * w + x;
                    let (a, b) = (src[i-1] as i32, src[i] as i32);
                    if (a - b).abs() <= chdb { pl[i-1] = js_round((3 * a + b) as f64 / 4.0) as u8; pl[i] = js_round((a + 3 * b) as f64 / 4.0) as u8; }
                    x += cs; } }
                // горизонтальные границы ячеек
                for x in 0..w { let mut y = cs; while y < h { let i = y * w + x;
                    let (a, b) = (src[i-w] as i32, src[i] as i32);
                    if (a - b).abs() <= chdb { pl[i-w] = js_round((3 * a + b) as f64 / 4.0) as u8; pl[i] = js_round((a + 3 * b) as f64 / 4.0) as u8; }
                    y += cs; } }
            }
        }
        // декодер-дизер (0 байт): Bayer-8×8 ±amp на ПЛОСКИХ зонах хромы — разбивает остаточные квант-ступени
        // в незаметный упорядоченный дизер (перцептивный дебэндинг). Края цвета (размах 3×3 > 8) не трогаем.
        // В проде — шейдер + blue-noise текстура (0-CPU). env MVQ_CHDITHER=amp (0=off).
        // ахроматизация тёмных (декод, 0 байт): чёрные линии ахроматичны, но хрома-bleeding через ÷2-границу
        // красит их пурпуром. Где luma_rec < darkt → тянем Cb/Cr к 128 пропорц. темноте (f=luma/darkt).
        // Тёмно-ЦВЕТНЫЕ (luma>darkt) не трогаем. env MVQ_CHDARK=порог (0=off).
        let darkt = envf("MVQ_CHDARK", 0.0);
        if darkt > 0.0 {
            let satt = envf("MVQ_CHDARK_SAT", 26.0); // не трогать НАСЫЩЕННОЕ тёмное (тёмно-красная тень) — только слабый паразит у линий
            for i in 0..n {
                let yl = luma_rec[i] as f64;
                let sat = (cbp[i] as f64 - 128.0).abs() + (crp[i] as f64 - 128.0).abs();
                if yl < darkt && sat < satt {
                    let f = yl / darkt; // 0 на чёрном → нейтраль, 1 на пороге → без изменений
                    cbp[i] = js_round(128.0 + (cbp[i] as f64 - 128.0) * f) as u8;
                    crp[i] = js_round(128.0 + (crp[i] as f64 - 128.0) * f) as u8;
                }
            }
        }
        let dith = envf("MVQ_CHDITHER", 0.0);
        if dith > 0.0 {
            static BAYER8: [i32; 64] = [
                0,48,12,60,3,51,15,63, 32,16,44,28,35,19,47,31, 8,56,4,52,11,59,7,55, 40,24,36,20,43,27,39,23,
                2,50,14,62,1,49,13,61, 34,18,46,30,33,17,45,29, 10,58,6,54,9,57,5,53, 42,26,38,22,41,25,37,21];
            for pl in [&mut cbp, &mut crp] {
                let src = pl.clone();
                for y in 1..h-1 { for x in 1..w-1 {
                    let i = y * w + x;
                    let (mut mn, mut mx) = (255i32, 0i32);
                    for dy in -1i32..=1 { for dx in -1i32..=1 { let v = src[(i as i32 + dy * w as i32 + dx) as usize] as i32; if v < mn { mn = v; } if v > mx { mx = v; } } }
                    if mx - mn <= 8 {
                        let b = BAYER8[(y & 7) * 8 + (x & 7)] as f64 / 64.0 - 0.5;
                        pl[i] = clamp_i(src[i] as i32 + js_round(b * dith)) as u8;
                    }
                }}
            }
        }
        Some((cbp, crp))
    } else { None };
    ChromaOut { k: k as f32, parts, planes, coarse: coarse_dump }
}

// порог planar-fit ошибки кэшируется (OnceLock): mode8_of зовётся на КАЖДЫЙ 8×8 блок люмы →
// envf (getenv+alloc) в нём = ~1.6M системных вызовов/главу. Читаем env один раз.
fn planar_me() -> f64 { static Q: std::sync::OnceLock<f64> = std::sync::OnceLock::new(); *Q.get_or_init(|| std::env::var("MVQ_PLANAR_ME").ok().and_then(|v| v.parse().ok()).unwrap_or(6.0)) }
// Порог planar-остатка для тёмных блоков: пускать тёмный ГРАДИЕНТ в pl8 (плоскость описывает хорошо, C0-гладко),
// исключать только тёмное ЗЕРНО (плоскость плохо, pl8_me велик → split4). default 0 = старое поведение (всё тёмное+aad вне pl8).
fn pl8_darkres() -> f64 { static Q: std::sync::OnceLock<f64> = std::sync::OnceLock::new(); *Q.get_or_init(|| std::env::var("MVQ_PL8_DARKRES").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0)) }
// mode8-классификация БЕЗ энкода (для сбора кодбука; зеркало JS mode8)
fn mode8_of(y_plane: &[u8], w: usize, h: usize, x0: usize, y0: usize) -> u8 {
    let inner = x0 + 8 <= w && y0 + 8 <= h;
    let mut lo = 255i32; let mut hi = 0i32; let mut isum = 0u32;
    if inner {
        for y in 0..8 {
            let row = &y_plane[(y0 + y) * w + x0..(y0 + y) * w + x0 + 8];
            for &v in row { let v = v as i32; if v < lo { lo = v; } if v > hi { hi = v; } isum += v as u32; }
        }
    } else {
        for y in 0..8 { for x in 0..8 { let v = y_plane[(y0 + y).min(h - 1) * w + (x0 + x).min(w - 1)] as i32; if v < lo { lo = v; } if v > hi { hi = v; } isum += v as u32; } }
    }
    let sum = isum as f64; // суммы целые → бит-точно прежнему f64-аккуму
    let mean = sum / 64.0;
    let mut vr = 0.0f64; let mut ext = 0i32;
    if inner {
        for y in 0..8 {
            let row = &y_plane[(y0 + y) * w + x0..(y0 + y) * w + x0 + 8];
            for &v in row { let v = v as i32; vr += (v as f64 - mean) * (v as f64 - mean); if v < lo + 40 || v > hi - 40 { ext += 1; } }
        }
    } else {
        for y in 0..8 { for x in 0..8 { let v = y_plane[(y0 + y).min(h - 1) * w + (x0 + x).min(w - 1)] as i32; vr += (v as f64 - mean) * (v as f64 - mean); if v < lo + 40 || v > hi - 40 { ext += 1; } } }
    }
    let sd = (vr / 64.0).sqrt();
    if hi - lo <= 2 { return 0; }
    let pf = if inner { plane_fit8_inner(y_plane, w, x0, y0) } else {
        let get = |x: usize, y: usize| -> u8 { y_plane[(y0 + y).min(h - 1) * w + (x0 + x).min(w - 1)] };
        plane_fit(&get, 8)
    };
    if pf.me <= planar_me() && !(hi <= 90 && sd >= 1.2 && pf.me > pl8_darkres()) { return 1; } // dark-исключение по ЗЕРНУ (pf.me велик), не яркости: гладкий тёмный градиент → planar8
    if sd >= 22.0 && sd <= 115.0 && (ext as f64) / 64.0 < 0.72 { return 2; }
    3
}
// сбор кодбука: параллельно по кадрам (локальные счётчики в порядке первой встречи),
// слияние в порядке кадров, stable-сорт по частоте — бит-точное зеркало JS collectCB
// FxHash: мультипликативный хэш вместо SipHash (крипто-стойкость не нужна; порядок карт не используется)
#[derive(Default, Clone)]
pub struct FxHasher { h: u64 }
impl std::hash::Hasher for FxHasher {
    #[inline] fn finish(&self) -> u64 { let h = self.h; h ^ (h >> 32) } // xor-shift: у мультипликативного хэша сильны верхние биты, ведро берёт нижние
    #[inline] fn write(&mut self, bytes: &[u8]) { for &b in bytes { self.h = (self.h.rotate_left(5) ^ b as u64).wrapping_mul(0x517cc1b727220a95); } }
    #[inline] fn write_u64(&mut self, i: u64) { self.h = (self.h.rotate_left(5) ^ i).wrapping_mul(0x517cc1b727220a95); }
}
#[derive(Default, Clone)]
pub struct FxBuild;
impl std::hash::BuildHasher for FxBuild { type Hasher = FxHasher; #[inline] fn build_hasher(&self) -> FxHasher { FxHasher::default() } }
type FxMap = std::collections::HashMap<u64, u32, FxBuild>;
type Scan = (Vec<u64>, FxMap, Vec<u8>);
fn cb_scan(w: usize, h: usize, y_plane: &[u8]) -> Scan {
    let est = (w / 8) * (h / 8); // верхняя оценка уников: 2 половины × доля mode3, ~половина блоков
    let mut order: Vec<u64> = Vec::with_capacity(est);
    let mut counts: FxMap = FxMap::with_capacity_and_hasher(est, FxBuild);
    let cw8 = (w + 7) / 8; let ch8 = (h + 7) / 8;
    let mut m8map = vec![255u8; cw8 * ch8]; // кэш классов для encode (255 = частичный блок, не сканирован)
    let bw8 = w / 8; let bh8 = h / 8;
    for by in 0..bh8 { for bx in 0..bw8 {
        let x0 = bx * 8; let y0 = by * 8;
        let mc = mode8_of(y_plane, w, h, x0, y0);
        m8map[by * cw8 + bx] = mc;
        if mc != 3 { continue; }
        let inner = x0 + 8 <= w && y0 + 8 <= h;
        for sy in 0..2usize { for sx in 0..2usize {
            let mut vv = [0u8; 16];
            if inner {
                let base = (y0 + sy * 4) * w + x0 + sx * 4;
                for row in 0..4 { vv[row * 4..row * 4 + 4].copy_from_slice(&y_plane[base + row * w..base + row * w + 4]); }
            } else {
                for k in 0..16 { vv[k] = y_plane[(y0 + sy * 4 + (k >> 2)).min(h - 1) * w + (x0 + sx * 4 + (k & 3)).min(w - 1)]; }
            }
            let (mut lo, mut hi) = (255u8, 0u8);
            for k in 0..16 { if vv[k] < lo { lo = vv[k]; } if vv[k] > hi { hi = vv[k]; } }
            if hi - lo <= 2 { continue; }
            let pf = plane_fit4(&vv); if pf.me <= 6.0 { continue; }
            for half in 0..2 {
                let key = u64::from_le_bytes(vv[half * 8..half * 8 + 8].try_into().unwrap());
                match counts.get_mut(&key) { Some(c) => *c += 1, None => { counts.insert(key, 1); order.push(key); } }
            }
        }}
    }}
    (order, counts, m8map)
}
fn collect_cb(views: &[(usize, usize, &[u8])], mut pre: Vec<Option<Scan>>, nth: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let nf = views.len();
    let mut locals: Vec<Option<Scan>> = (0..nf).map(|_| None).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let next_ref = &next; let views_ref = views; let pre_ref = &pre;
    std::thread::scope(|sc| {
        let mut hs = Vec::new();
        for _ in 0..nth.min(nf.max(1)) {
            hs.push(sc.spawn(move || {
                let mut out: Vec<(usize, Scan)> = Vec::new();
                loop {
                    let fi = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if fi >= nf { break; }
                    if pre_ref[fi].is_some() { continue; } // уже посчитано в декод-пассе
                    let (w, h, y_plane) = views_ref[fi];
                    out.push((fi, cb_scan(w, h, y_plane)));
                }
                out
            }));
        }
        for hd in hs { for (fi, r) in hd.join().unwrap() { locals[fi] = Some(r); } }
    });
    // шардированный мердж (8 потоков): глобальный ранг первой встречи = (fi, pos) — вместе с count
    // воспроизводит V8-stable-sort-по-insertion-order бит-в-бит: sort(count desc, fi asc, pos asc)
    let fdata: Vec<Scan> = locals.into_iter().enumerate()
        .map(|(fi, l)| match l { Some(v) => v, None => pre[fi].take().unwrap() }).collect();
    // (шардовые карты ниже тоже FxHash)
    const SH: usize = 16;
    let fdata_ref = &fdata;
    let mut shards: Vec<Vec<(u64, u32, u32, u32)>> = Vec::with_capacity(SH);
    std::thread::scope(|sc| {
        let mut hs = Vec::new();
        for shard in 0..SH {
            hs.push(sc.spawn(move || {
                let mut m: std::collections::HashMap<u64, (u32, u32, u32), FxBuild> = std::collections::HashMap::default();
                for (fi, (order, counts, _)) in fdata_ref.iter().enumerate() {
                    for (pos, &key) in order.iter().enumerate() {
                        if (key as usize) & (SH - 1) != shard { continue; }
                        let c = counts[&key];
                        match m.get_mut(&key) {
                            Some(e) => e.0 += c,
                            None => { m.insert(key, (c, fi as u32, pos as u32)); }
                        }
                    }
                }
                m.into_iter().map(|(k, (c, fi, pos))| (k, c, fi, pos)).collect::<Vec<_>>()
            }));
        }
        for hd in hs { shards.push(hd.join().unwrap()); }
    });
    let mut ent: Vec<(u64, u32, u32, u32)> = shards.into_iter().flatten().collect();
    // отсев count==1 (не попадут в топ-256, если кандидатов с count>=2 достаточно)
    let multi = ent.iter().filter(|e| e.1 >= 2).count();
    if multi >= 256 { ent.retain(|e| e.1 >= 2); }
    ent.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3)));
    let ke = ent.len().min(256); // SUBK=256
    let mut cb = Vec::with_capacity(ke * 8);
    for i in 0..ke { cb.extend_from_slice(&ent[i].0.to_le_bytes()); }
    (cb, fdata.into_iter().map(|s| s.2).collect())
}

// Гибрид-LCM (2026-07-06, матрица per-stream): 83% весовой пользы контекст-моделинга сидит в 8 типах
// потоков (m8/pc/dsg/dmgM/dmgV/sub/cov/rz0v) за ~30% его времени. Прочим — noLCM.
// MVQ_LCM=full — везде (архив); MVQ_LCM=off — нигде (спидран, +2.1%).
fn lcm_mode() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| match std::env::var("MVQ_LCM").as_deref() {
        Ok("full") => 2,
        Ok("off") => 0,
        _ => 1,
    })
}
#[inline] fn lcm_for_luma(idx: usize) -> bool {
    matches!(idx, 0 | 2 | 5 | 6 | 7 | 9 | 14 | 21) // m8, pc, dsg, dmgM, dmgV, sub, cov, rz0v
}
fn br_q(data: &[u8], q: i32) -> Vec<u8> { br_q_lcm(data, q, true) }
fn br_q_lcm(data: &[u8], q: i32, lcm: bool) -> Vec<u8> {
    if data.is_empty() { return Vec::new(); }
    let mut out = Vec::new();
    let mut params = brotli::enc::BrotliEncoderParams::default();
    params.quality = q;
    params.size_hint = data.len();
    let use_lcm = match lcm_mode() { 2 => true, 0 => false, _ => lcm };
    if !use_lcm { params.disable_literal_context_modeling = 1; }
    // окно = размер входа (кламп 10..22): вес идентичен (окно >= входа), мелкие потоки не платят за 4МБ хэш-таблиц
    let mut lg = 10i32;
    while (1usize << lg) < data.len() && lg < 22 { lg += 1; }
    params.lgwin = lg;
    // mvq: пропускаем distance-prefix свип (npostfix×ndirect, до 4×16 полных command-проходов +
    // PopulationCost на метаблок). Для бинарных потоков победитель почти всегда postfix=0/ndirect=0.
    params.avoid_distance_prefix_search = true;
    brotli::BrotliCompress(&mut &data[..], &mut out, &params).unwrap();
    out
}

// ---- декод входа (paths-режим, bit25): jpeg=zune, webp=image-webp; grey-jpeg → ch=1 (как sharp) ----
struct Decoded { w: usize, h: usize, ch: usize, rgb: Vec<u8>, luma: Vec<u8>, mh: bool }
fn luma_of(data: &[u8], w: usize, h: usize, ch: usize) -> Vec<u8> {
    // целочисленный BT.601: (19595r+38470g+7471b+32768)>>16; Σкоэфф=65536 → результат ≤255 без клампа.
    // Дрейф ±1 на редких пикселях vs js_round-float; JS-раннер синхронизирован 2026-07-06.
    let mut y = vec![0u8; w * h];
    if ch == 3 {
        for (yp, px) in y.iter_mut().zip(data.chunks_exact(3)) {
            *yp = ((19595 * px[0] as u32 + 38470 * px[1] as u32 + 7471 * px[2] as u32 + 32768) >> 16) as u8;
        }
    } else if ch >= 3 {
        for (yp, px) in y.iter_mut().zip(data.chunks_exact(ch)) {
            *yp = ((19595 * px[0] as u32 + 38470 * px[1] as u32 + 7471 * px[2] as u32 + 32768) >> 16) as u8;
        }
    } else { y.copy_from_slice(&data[..w * h]); }
    y
}
fn mh_of(data: &[u8], w: usize, h: usize, ch: usize) -> bool {
    if ch < 3 { return false; }
    let (mut colorful, mut tot) = (0u32, 0u32);
    let mut p = 0usize;
    while p < w * h {
        let o = p * ch;
        let (r, g, b) = (data[o] as i32, data[o + 1] as i32, data[o + 2] as i32);
        let d = (r - g).abs().max((g - b).abs()).max((r - b).abs());
        if d > 12 { colorful += 1; }
        tot += 1;
        p += 4;
    }
    tot > 0 && colorful as f64 / tot as f64 > 0.10
}
pub static DEC_PROF: [core::sync::atomic::AtomicU64; 4] = [
    core::sync::atomic::AtomicU64::new(0), // 0 = file read + jpeg/webp decode
    core::sync::atomic::AtomicU64::new(0), // 1 = luma_of
    core::sync::atomic::AtomicU64::new(0), // 2 = mh_of
    core::sync::atomic::AtomicU64::new(0), // 3 = cb_scan
];
fn decode_file(path: &str) -> Option<Decoded> {
    let __t = std::time::Instant::now();
    let data = std::fs::read(path).ok()?;
    let lower = path.to_lowercase();
    let (w, h, ch, rgb): (usize, usize, usize, Vec<u8>) = if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        use zune_jpeg::JpegDecoder;
        use zune_core::colorspace::ColorSpace;
        use zune_core::options::DecoderOptions;
        let mut dec = JpegDecoder::new_with_options(std::io::Cursor::new(&data[..]), DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB));
        dec.decode_headers().ok()?;
        let grey = matches!(dec.input_colorspace(), Some(ColorSpace::Luma));
        if grey {
            let mut d2 = JpegDecoder::new_with_options(std::io::Cursor::new(&data[..]), DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::Luma));
            let px = d2.decode().ok()?;
            let (w, h) = d2.dimensions()?;
            (w as usize, h as usize, 1, px)
        } else {
            let px = dec.decode().ok()?;
            let (w, h) = dec.dimensions()?;
            (w as usize, h as usize, 3, px)
        }
    } else if lower.ends_with(".webp") {
        let cur = std::io::Cursor::new(&data[..]);
        let dec = image_webp::WebPDecoder::new(cur).ok()?;
        let (w, h) = dec.dimensions();
        let mut dec = dec;
        let has_a = dec.has_alpha();
        let ch = if has_a { 4 } else { 3 };
        let mut px = vec![0u8; w as usize * h as usize * ch];
        dec.read_image(&mut px).ok()?;
        (w as usize, h as usize, ch, px)
    } else { return None; };
    DEC_PROF[0].fetch_add(__t.elapsed().as_micros() as u64, core::sync::atomic::Ordering::Relaxed);
    let __t1 = std::time::Instant::now();
    let luma = luma_of(&rgb, w, h, ch);
    DEC_PROF[1].fetch_add(__t1.elapsed().as_micros() as u64, core::sync::atomic::Ordering::Relaxed);
    let __t2 = std::time::Instant::now();
    let mh = mh_of(&rgb, w, h, ch);
    DEC_PROF[2].fetch_add(__t2.elapsed().as_micros() as u64, core::sync::atomic::Ordering::Relaxed);
    Some(Decoded { w, h, ch, rgb: if ch >= 3 { rgb } else { Vec::new() }, luma, mh })
}

// MVQ_NTH: оверрайд числа потоков (SMT-эксперименты: физядра vs 2×SMT)
fn nth_conf() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("MVQ_NTH").ok().and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)))
}
fn rd32(b: &[u8], o: usize) -> usize { u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize }

fn main() {
    init_bits_lut();
    let __w0 = std::time::Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let mut buf = Vec::new();
    File::open(&args[1]).unwrap().read_to_end(&mut buf).unwrap();
    let n_frames = rd32(&buf, 0);
    let ke_in = rd32(&buf, 4);
    let collect_mode = ke_in == 0xFFFF_FFFF;
    let ke_hdr = if collect_mode { 0 } else { ke_in };
    let cb_hdr = buf[8..8 + ke_hdr * 8].to_vec();
    let mut off = 8 + ke_hdr * 8;
    // (w,h,luma_off,mh,mhdct,qpreset,fast,emit_planes,rgb_off,ch); rgb_off=usize::MAX если хромы нет
    // флаги: bit0 mh | bits8-15 mhdct | bits16-18 qpreset | bit19 fast | bit21 emit_planes | bit24 rgb приложен
    let mut frames: Vec<(usize, usize, usize, bool, i32, usize, bool, bool, usize, usize)> = Vec::new();
    let mut skip_rec = false; let mut br_rust = false;
    let mut paths: Vec<String> = vec![String::new(); n_frames];
    let mut any_path = false;
    for fi in 0..n_frames {
        let w = rd32(&buf, off); let h = rd32(&buf, off + 4); let flags = rd32(&buf, off + 8);
        let is_path = flags & (1 << 25) != 0;
        if is_path {
            let plen = u16::from_le_bytes([buf[off + 12], buf[off + 13]]) as usize;
            paths[fi] = String::from_utf8_lossy(&buf[off + 14..off + 14 + plen]).to_string();
            any_path = true;
            off += 14 + plen;
            frames.push((0, 0, usize::MAX, false, ((flags >> 8) & 0xff) as i32, (flags >> 16) & 7,
                         flags & (1 << 19) != 0, flags & (1 << 21) != 0, usize::MAX, 0));
        } else {
            let has_rgb = flags & (1 << 24) != 0;
            let luma_off = off + 12;
            off = luma_off + w * h;
            let (rgb_off, ch) = if has_rgb {
                let c = buf[off] as usize; let ro = off + 1; off = ro + w * h * c; (ro, c)
            } else { (usize::MAX, 0) };
            frames.push((w, h, luma_off, flags & 1 == 1, ((flags >> 8) & 0xff) as i32, (flags >> 16) & 7,
                         flags & (1 << 19) != 0, flags & (1 << 21) != 0, rgb_off, ch));
        }
        skip_rec = flags & (1 << 22) != 0; br_rust = flags & (1 << 23) != 0;
    }
    // декод-пасс (параллельный) для path-кадров (+fuse: cb-скан там же)
    let mut owned: Vec<Option<Decoded>> = (0..n_frames).map(|_| None).collect();
    let mut cb_pre: Vec<Option<Scan>> = (0..n_frames).map(|_| None).collect();
    if any_path {
        let nth_d = nth_conf().min(n_frames.max(1));
        let next_d = std::sync::atomic::AtomicUsize::new(0);
        let next_ref = &next_d; let paths_ref = &paths;
        let mut slots: Vec<Option<Decoded>> = (0..n_frames).map(|_| None).collect();
        let mut pre_slots: Vec<Option<Scan>> = (0..n_frames).map(|_| None).collect();
        thread::scope(|sc| {
            let mut hs = Vec::new();
            for _ in 0..nth_d {
                hs.push(sc.spawn(move || {
                    let mut out: Vec<(usize, Decoded, Scan)> = Vec::new();
                    loop {
                        let fi = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if fi >= n_frames { break; }
                        if paths_ref[fi].is_empty() { continue; }
                        match decode_file(&paths_ref[fi]) {
                            Some(d) => { let __t3 = std::time::Instant::now(); let sc2 = cb_scan(d.w, d.h, &d.luma); DEC_PROF[3].fetch_add(__t3.elapsed().as_micros() as u64, core::sync::atomic::Ordering::Relaxed); out.push((fi, d, sc2)); }
                            None => { eprintln!("mvq: DECODE FAIL {}", paths_ref[fi]); std::process::exit(3); }
                        }
                    }
                    out
                }));
            }
            for hd in hs { for (fi, d, sc2) in hd.join().unwrap() { slots[fi] = Some(d); pre_slots[fi] = Some(sc2); } }
        });
        owned = slots;
        cb_pre = pre_slots;
        for fi in 0..n_frames {
            if owned[fi].is_some() {
                let (dw, dh, dch, dmh) = { let d = owned[fi].as_ref().unwrap(); (d.w, d.h, d.ch, d.mh) };
                let f = &mut frames[fi];
                f.0 = dw; f.1 = dh; f.3 = dmh;
                f.9 = dch;
                f.8 = usize::MAX - 1; // path-кадры ВСЕГДА через encode_chroma (ch<3 → 128-планы; формат требует хрома-блок)
            }
        }
    }
    let owned_ref = &owned;
    let __w_dec = __w0.elapsed().as_secs_f64();
    let mut c8 = [[0.0f64; 8]; 8];
    for u in 0..8 { for x in 0..8 {
        c8[u][x] = ((2.0 * x as f64 + 1.0) * u as f64 * std::f64::consts::PI / 16.0).cos() * if u == 0 { (1.0f64 / 8.0).sqrt() } else { (2.0f64 / 8.0).sqrt() };
    }}
    let nth = nth_conf().min(n_frames.max(1));
    let luma_views: Vec<(usize, usize, &[u8])> = frames.iter().enumerate().map(|(i, f)| {
        (f.0, f.1, if f.2 == usize::MAX { &owned_ref[i].as_ref().unwrap().luma[..] } else { &buf[f.2..f.2 + f.0 * f.1] })
    }).collect();
    let luma_views_ref = &luma_views;
    let __tcb = std::time::Instant::now();
    let (cb, m8maps) = if collect_mode { collect_cb(&luma_views, cb_pre, nth) } else { (cb_hdr, Vec::new()) };
    let m8maps_ref = &m8maps;
    let ke = cb.len() / 8;
    let t_cb = __tcb.elapsed().as_secs_f64();
    let mut results: Vec<Option<(Vec<Vec<u8>>, f64, Vec<u8>, Option<ChromaOut>, Option<Vec<u8>>)>> = (0..n_frames).map(|_| None).collect();
    let buf_ref = &buf; let cb_ref = &cb; let c8_ref = &c8; let frames_ref = &frames;
    // work-stealing: тяжёлые кадры первыми, атомарный указатель вместо статических чанков
    let mut order: Vec<usize> = (0..n_frames).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(frames[i].0 * frames[i].1));
    let order_ref = &order;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let next_ref = &next;
    let tl_acc = std::sync::atomic::AtomicUsize::new(0); let tl_ref = &tl_acc;
    let tc_acc = std::sync::atomic::AtomicUsize::new(0); let tc_ref = &tc_acc;
    let tb_acc = std::sync::atomic::AtomicUsize::new(0); let tb_ref = &tb_acc;
    let brq: i32 = std::env::var("MVQ_BRQ").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let mq9: bool = std::env::var("MVQ_MASKQ9").is_ok(); // q9 на больших масках: −2мс/стр за +0.8% веса (опция)
    let _ = brq; let skip_rec_c = skip_rec; let br_rust_c = br_rust; let _ = (skip_rec_c, br_rust_c);
    let __wall = std::time::Instant::now();
    thread::scope(|sc| {
        let mut handles = Vec::new();
        for _t in 0..nth {
            handles.push(sc.spawn(move || {
                let mut out: Vec<(usize, (Vec<Vec<u8>>, f64, Vec<u8>, Option<ChromaOut>, Option<Vec<u8>>))> = Vec::new();
                loop {
                    let k = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if k >= n_frames { break; }
                    let fi = order_ref[k];
                    let (w, h, _o, mh, mhdct, qp, fast, emit, rgb_off, ch) = frames_ref[fi];
                    let y = luma_views_ref[fi].2;
                    let __tl = std::time::Instant::now();
                    let m8s: Option<&[u8]> = m8maps_ref.get(fi).map(|v| &v[..]);
                    let (mut parts, sse, rec) = encode_frame(y, w, h, cb_ref, ke, c8_ref, mh, mhdct, qp, fast, m8s);
                    tl_ref.fetch_add(__tl.elapsed().as_micros() as usize, std::sync::atomic::Ordering::Relaxed);
                    let __tc = std::time::Instant::now();
                    let mut chroma = if rgb_off == usize::MAX - 1 {
                        Some(encode_chroma(&owned_ref[fi].as_ref().unwrap().rgb[..], ch, w, h, &rec, emit))
                    } else if rgb_off != usize::MAX {
                        Some(encode_chroma(&buf_ref[rgb_off..rgb_off + w * h * ch], ch, w, h, &rec, emit))
                    } else { None };
                    tc_ref.fetch_add(__tc.elapsed().as_micros() as usize, std::sync::atomic::Ordering::Relaxed);
                    if br_rust {
                        let __tb = std::time::Instant::now();
                        // битмаски (dmg_mask=6, rz-маски 20/23/26/29) жмём q9: их q9-цена +2.2% от маски (~+0.3% общего),
                        // время −70%; формат не меняется (brotli есть brotli)
                        for (i, p) in parts.iter_mut().enumerate() {
                            let is_mask = i == 6 || i == 20 || i == 23 || i == 26 || i == 29;
                            let q = if mq9 && is_mask && p.len() > 16384 { brq.min(9) } else { brq };
                            *p = br_q_lcm(p, q, lcm_for_luma(i));
                        }
                        if let Some(co) = chroma.as_mut() { for p in co.parts.iter_mut() { *p = br_q_lcm(p, brq, true); } } // хрома мелкая — LCM вся
                        tb_ref.fetch_add(__tb.elapsed().as_micros() as usize, std::sync::atomic::Ordering::Relaxed);
                    }
                    out.push((fi, (parts, sse, rec, chroma, m8s.map(|s| s.to_vec()))));
                }
                out
            }));
        }
        for hd in handles {
            for (fi, r) in hd.join().unwrap() { results[fi] = Some(r); }
        }
    });
    let mut out: Vec<u8> = Vec::new();
    let structdump = std::env::var("MVQ_STRUCTDUMP").is_ok(); // диагностика: дописать coarse-recon хромы за planes
    if collect_mode { out.extend_from_slice(&(ke as u32).to_le_bytes()); out.extend_from_slice(&cb); }
    for (fi, r) in results.into_iter().enumerate() {
        let (parts, sse, rec, chroma, m8dump) = r.unwrap();
        if !paths[fi].is_empty() {
            let f = &frames[fi];
            out.extend_from_slice(&(f.0 as u32).to_le_bytes());
            out.extend_from_slice(&(f.1 as u32).to_le_bytes());
            out.extend_from_slice(&(if f.3 { 1u32 } else { 0u32 }).to_le_bytes());
        }
        out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
        for p in &parts { out.extend_from_slice(&(p.len() as u32).to_le_bytes()); out.extend_from_slice(p); }
        out.extend_from_slice(&sse.to_le_bytes());
        if !skip_rec { out.extend_from_slice(&rec); }
        if let Some(co) = chroma {
            out.extend_from_slice(&co.k.to_le_bytes());
            out.extend_from_slice(&(co.parts.len() as u32).to_le_bytes());
            for p in &co.parts { out.extend_from_slice(&(p.len() as u32).to_le_bytes()); out.extend_from_slice(p); }
            if let Some((cbp, crp)) = co.planes { out.extend_from_slice(&cbp); out.extend_from_slice(&crp); }
            if structdump { if let Some((ca, cr2, sw2, sh2)) = co.coarse {
                out.extend_from_slice(&sw2.to_le_bytes()); out.extend_from_slice(&sh2.to_le_bytes());
                out.extend_from_slice(&ca); out.extend_from_slice(&cr2);
            } }
        }
        if structdump { // m8map (mode per 8×8 люмы) для структурного люма-banding детектора
            let cw8 = ((frames[fi].0 + 7) / 8) as u32; let ch8 = ((frames[fi].1 + 7) / 8) as u32;
            out.extend_from_slice(&cw8.to_le_bytes()); out.extend_from_slice(&ch8.to_le_bytes());
            if let Some(m) = m8dump { out.extend_from_slice(&m); } else { out.extend(std::iter::repeat(255u8).take((cw8 * ch8) as usize)); }
        }
        let _ = fi;
    }
    let __w_pool = __w0.elapsed().as_secs_f64();
    File::create(&args[2]).unwrap().write_all(&out).unwrap();
    if std::env::var("MVQ_BRPROF").is_ok() {
        use core::sync::atomic::Ordering;
        let d = &DEC_PROF;
        eprintln!("DECPROF: file+decode {:.1} | luma_of {:.1} | mh_of {:.1} | cb_scan {:.1} core-s || WALL: dec-pass {:.2}s, до-write {:.2}s, write+exit {:.2}s",
            d[0].load(Ordering::Relaxed) as f64/1e6, d[1].load(Ordering::Relaxed) as f64/1e6,
            d[2].load(Ordering::Relaxed) as f64/1e6, d[3].load(Ordering::Relaxed) as f64/1e6,
            __w_dec, __w_pool - __w_dec, __w0.elapsed().as_secs_f64() - __w_pool);
    }
    if std::env::var("MVQ_BRPROF").is_ok() {
        use core::sync::atomic::Ordering;
        let p = &brotli::enc::encode::BR_PROF;
        {
            let c = &CH_PROF;
            eprintln!("CHPROF: planes {:.1} | quad {:.1} | guided {:.1} | cgain {:.1} | patches {:.1} core-s",
                c[0].load(core::sync::atomic::Ordering::Relaxed) as f64/1e6, c[1].load(core::sync::atomic::Ordering::Relaxed) as f64/1e6,
                c[2].load(core::sync::atomic::Ordering::Relaxed) as f64/1e6, c[3].load(core::sync::atomic::Ordering::Relaxed) as f64/1e6,
                c[4].load(core::sync::atomic::Ordering::Relaxed) as f64/1e6);
        }
        eprintln!("BRPROF: backward_refs {:.1}s | build_metablock {:.1}s | store_metablock {:.1}s",
            p[0].load(Ordering::Relaxed) as f64/1e6, p[1].load(Ordering::Relaxed) as f64/1e6, p[2].load(Ordering::Relaxed) as f64/1e6);
    }
    eprintln!("mvq: {} frames done | cb {:.2}s | luma {:.1} chroma {:.1} brotli {:.1} core-s, wall {:.1}s, nth {}",
        n_frames, t_cb, tl_acc.load(std::sync::atomic::Ordering::Relaxed) as f64/1e6, tc_acc.load(std::sync::atomic::Ordering::Relaxed) as f64/1e6, tb_acc.load(std::sync::atomic::Ordering::Relaxed) as f64/1e6, __wall.elapsed().as_secs_f64(), nth);
}
