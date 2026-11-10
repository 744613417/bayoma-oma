function test_py_case()
%TEST_PY_CASE 用 Python 侧生成的同一份数据校核 MATLAB 多模态估计器。
%
%   目的是把"数据合成"与"估计器实现"分开：同一份缩放 FFT 分别送入两套估计器，
%   若结果一致，则差异来自各自的数据合成环节而非估计器实现。
%
%   数据文件 py_multi_case.mat 不随工程分发，缺失时调用 Python 侧脚本重新生成。
%   该脚本要求 Python 侧包已在搜索路径上（由环境变量 `BAYOMA_PYTHON` 指定，缺省 `python`）。

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

here = fileparts(mfilename('fullpath'));
matPath = fullfile(here, 'py_multi_case.mat');
if exist(matPath, 'file') ~= 2
    genScript = fullfile(fileparts(here), '..', 'python', 'tests', 'gen_cross_case.py');
    if exist(genScript, 'file') ~= 2
        fprintf('缺少 %s，且生成脚本 %s 不存在，跳过本项校核。\n', matPath, genScript);
        return;
    end
    pyExe = getenv('BAYOMA_PYTHON');
if isempty(pyExe)
    pyExe = 'python';
end
cmd = sprintf('"%s" "%s"', pyExe, genScript);
    [status, msg] = system(cmd);
    if status ~= 0
        fprintf('生成脚本执行失败（%d）：%s\n', status, msg);
        return;
    end
end

s = load(matPath);
fft = s.fft;
freqs = s.freqs(:);
dTotal = s.dTotal;
fTrue = s.f(:);
zTrue = s.z(:);
phiTrue = s.phi;
STrue = s.S;

nch = size(fft, 1);
nf = size(fft, 2);
D = zeros(nch, nch, nf);
for k = 1:nf
    fk = fft(:, k);
    D(:, :, k) = real(fk * fk');
end

opts = struct('f0', fTrue, 'q', 0, 'maxiter', 40);
out = bayoma.multi_mode(D, fft, freqs, 2, opts);

fprintf('MATLAB f   = [%.6f %.6f]  真值 [%.4f %.4f]\n', out.f(1), out.f(2), fTrue(1), fTrue(2));
fprintf('MATLAB z   = [%.6f %.6f]  真值 [%.4f %.4f]\n', out.z(1), out.z(2), zTrue(1), zTrue(2));
fprintf('MATLAB Se  = %.6f  真值 %.4f\n', out.Se, s.Se);
fprintf('MATLAB S12 = %.4f%+.4fj  真值 %.4f%+.4fj\n', ...
    real(out.S(1,2)), imag(out.S(1,2)), real(STrue(1,2)), imag(STrue(1,2)));
mac = abs(sum(conj(out.phi) .* phiTrue, 1));
fprintf('MATLAB MAC = [%.6f %.6f]  niter=%d conv=%d\n', mac(1), mac(2), out.niter, out.converged);
end
