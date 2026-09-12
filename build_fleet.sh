#!/bin/bash
# Флот-сборка mvq: ./build_fleet.sh {epyc7742|7950x3d|native}
case "$1" in
  epyc7742) CPU=znver2 ;;
  7950x3d)  CPU=znver4 ;;
  native|*) CPU=native ;;
esac
RUSTFLAGS="-C target-cpu=$CPU" cargo build --release
echo "built for $CPU"

# ============================ EPYC 7742 ×2 — ПЛЕЙБУК ============================
# ЖЕЛЕЗО: 2 сокета × 64C/128T Zen2; L3 = 16 CCX × 16МБ/сокет; 8 каналов DDR4/сокет.
#
# 1) BIOS: NPS4 (4 NUMA-домена на сокет; локальность памяти), Determinism = Performance,
#    cTDP/PPT в максимум платформы, SMT ON (наш замер: +35% throughput).
# 2) ОРКЕСТРАЦИЯ: 8 процессов mvq (по одному на NUMA-домен), каждому свой список глав:
#      numactl --cpunodebind=$N --membind=$N ./mvq in.bin out.bin   (MVQ_NTH=32)
#    Внутрипроцессный work-stealing останется в пределах домена — кросс-CCX трафик минимален.
#    Главы в списке сортировать по размеру убыванию (ровный финиш).
# 3) ВРЕМЕННЫЕ ФАЙЛЫ: in/out класть в /dev/shm (tmpfs) — файловый протокол без диска.
# 4) АЛЛОКАТОР: mimalloc (cargo add mimalloc + #[global_allocator]; на Linux +2-5%).
# 5) THP: echo always > /sys/kernel/mm/transparent_hugepage/enabled (+3-8% на TLB:
#    RGB-буферы, tprime, brotli-деревья — крупные аллокации).
# 6) PGO:  a) RUSTFLAGS="-C target-cpu=znver2 -Cprofile-generate=/tmp/pgo" cargo build --release
#          b) прогнать корпус (наш titles/ ок)   c) llvm-profdata merge -o /tmp/m.profdata /tmp/pgo
#          d) RUSTFLAGS="-C target-cpu=znver2 -Cprofile-use=/tmp/m.profdata" cargo build --release
#    Ожидание +5-15% (ветвистый brotli).
# 7) BOLT (опция поверх PGO): llvm-bolt на готовом бинаре с perf-профилем — ещё +5-10%
#    на больших ветвистых бинарях (перекладка код-лейаута).
# 8) mitigations=off в cmdline ядра (приватный конвертер-сервер): +1-3%.
# 9) Калибровка на месте: MVQ_NTH-кривая (24/28/32 на домен) + _selftest.js после сборки.
# СУММАРНОЕ ОЖИДАНИЕ поверх базовой оценки 14.4ч: −15..30% → ~10-12 часов на 17.66M.
#
# ============================ 7950X3D — ПЛЕЙБУК ============================
# znver4-сборка (выше). ОБЯЗАТЕЛЬНО: node _selftest.js (валидация AVX-512-ветки, TIER1==TIER2).
# Один NUMA-узел — numactl не нужен; MVQ_NTH-кривая (16/24/32); V-Cache сам подхватится.
