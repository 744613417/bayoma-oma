function test_select_bands()
%TEST_SELECT_BANDS 校核自动选带：分离模态与密集模态两种情形。
%
%   分离模态（1.0 / 1.4 Hz）应判出两个单阶频带；密集模态（1.00 / 1.02 Hz）
%   应判出一个两阶频带。噪声起伏不得被判为模态。

phi = [1; 2; 2] / 3;
fs = 100;
dur = 2400;

% 情形 1：分离模态
data = bayoma.simulate_oma_data([1.0, 1.4], [0.01, 0.01], phi, eye(2), 1.0, ...
    fs, dur, struct('seed', 5));
[freqs, psd] = bayoma.welch_psd(data, fs);
sel = bayoma.select_bands(freqs, psd);
fprintf('分离模态：peaks = %s，nmodes = %s\n', mat2str(sel.peaks.', 4), mat2str(sel.nmodes.'));
fprintf('           bands = %s\n', mat2str(sel.bands, 4));
assert(numel(sel.peaks) == 2, '分离模态应判出两个谱峰');
assert(all(abs(sel.peaks(:) - [1.0; 1.4]) < 0.05), '分离模态的谱峰频率偏离真值');
assert(all(sel.nmodes == 1), '分离模态各频带应为一阶');

% 情形 2：密集模态
coh = 0.5 * exp(1i * pi / 4);
S = [1.0, coh; conj(coh), 1.0];
phi2 = [[1; 2; 2] / 3, [-2; 1; 2] / 3];
data = bayoma.simulate_oma_data([1.0, 1.02], [0.01, 0.01], phi2, S, 1.0, ...
    fs, dur, struct('seed', 3));
[freqs, psd] = bayoma.welch_psd(data, fs);
sel = bayoma.select_bands(freqs, psd);
fprintf('密集模态：peaks = %s，nmodes = %s\n', mat2str(sel.peaks.', 4), mat2str(sel.nmodes.'));
fprintf('           bands = %s\n', mat2str(sel.bands, 4));
assert(numel(sel.peaks) == 1, '密集模态应判出一个谱峰');
assert(sel.nmodes(1) == 2, '密集模态的频带应判为两阶');

fprintf('自动选带校核通过\n');
end
