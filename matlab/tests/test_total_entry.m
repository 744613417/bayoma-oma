function test_total_entry()
%TEST_TOTAL_ENTRY 总入口的端到端校核：单模态频带、多模态频带、自动选带。

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

nch = 3;
fs = 100;
dur = 1200;
phiTrue = [1; 2; 2] / 3;
opts = struct('q', 0, 'seed', 7);
tdata = bayoma.simulate_oma_data(1.0, 0.01, phiTrue, 1.0, 1.0, fs, dur, opts);

IN = struct();
IN.tdata = tdata;
IN.fs = fs;
IN.f1f2 = [0.9 1.1];
IN.f0 = {1.0};
IN.mute = 1;
out = bayoma.bayoma(IN);
fprintf('单模态频带：f=%.6f z=%.6f Sii=%.4f Se=%.4f sn=%.1f\n', ...
    out.f, out.z, out.Sii, out.Se, out.sn);
fprintf('            phi = [%.5f %.5f %.5f]  coefv.f=%.6f coefv.phi=%.6f\n', ...
    out.phi, out.coefv.f, out.coefv.phi);
end
