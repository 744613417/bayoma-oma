function test_multi_mode()
%TEST_MULTI_MODE 多模态识别的合成数据校核。
%
%   采用课本例 13.2 的参数：两阶密集模态 f = [1, 1.02] Hz，模态力谱矩阵含
%   幅值 0.5、相位 pi/4 的互谱项。核对频率、阻尼比、模态力谱矩阵与振型。

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fs = 100;
dur = 2400;
f = [1.0; 1.02];
z = [0.01; 0.01];
phi = [1 -2; 2 1; 2 2] ./ 3;
phi = phi ./ sqrt(sum(phi.^2, 1));
S = [1.0, 0.5 * exp(1i * pi / 4); 0.5 * exp(-1i * pi / 4), 1.0];
Se = 1.0;
q = 0;

opts = struct('q', q, 'seed', 3);
tdata = bayoma.simulate_oma_data(f, z, phi, S, Se, fs, dur, opts);

Sspec = bayoma.scaled_fft(tdata, fs, [0.9 1.1]);
[D, ~] = bayoma.psd_matrices(Sspec);

mopts = struct('f0', f, 'q', q, 'maxiter', 40);
out = bayoma.multi_mode(D, Sspec.fft, Sspec.freqs, 2, mopts);

fprintf('f   = [%.6f %.6f]  (真值 [%.4f %.4f])\n', out.f(1), out.f(2), f(1), f(2));
fprintf('z   = [%.6f %.6f]  (真值 [%.4f %.4f])\n', out.z(1), out.z(2), z(1), z(2));
fprintf('Se  = %.6f  (真值 %.4f)\n', out.Se, Se);
fprintf('S   = [%.4f%+.4fi  %.4f%+.4fi]\n', ...
    real(out.S(1,1)), imag(out.S(1,1)), real(out.S(1,2)), imag(out.S(1,2)));
fprintf('   真值 [%.4f  %.4f%+.4fi]\n', real(S(1,1)), real(S(1,2)), imag(S(1,2)));
mac = abs(sum(conj(out.phi) .* phi, 1));
fprintf('MAC = [%.6f %.6f]   niter = %d   converged = %d\n', mac(1), mac(2), out.niter, out.converged);
end
