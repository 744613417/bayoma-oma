% 参考自由度凝聚校核：课本 Example 14.2
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..')));
nDof = 6;
sel = { selmat([6 3 4], nDof), selmat([6 3 2], nDof), selmat([5 2], nDof), selmat([5 1], nDof) };
til = { [-0.642; -0.454; -0.618], [0.804; 0.420; 0.421], [0.883; 0.470], [0.959; 0.284] };

[Lc, info] = bayoma.condensation_matrix({[6 3 4], [6 3 2], [5 2], [5 1]}, nDof);
fprintf('reference dofs = %s  nCondensed = %d  orth err = %.3e\n', ...
        mat2str(info.reference), info.nCondensed, max(max(abs(Lc'*Lc - eye(size(Lc,2))))));

expect = [0.162; 0.286; 0.315; 0.477; 0.546; 0.516];
expect = expect / norm(expect);
for cond = [false, true]
    out = bayoma.global_least_squares(til, sel, [], struct('condense', cond));
    u = out.u;
    if u(1) < 0
        u = -u;
    end
    mac = (u'*expect)^2 / ((u'*u)*(expect'*expect));
    fprintf('condense=%d  nIter=%d conv=%d  MAC=%.6f  maxdiff=%.3e\n', ...
            cond, out.nIter, out.converged, mac, max(abs(u - expect)));
    fprintf('   u = [%s]\n', sprintf('%.4f ', u));
end
fprintf('   expect = [%s]\n', sprintf('%.4f ', expect));

function m = selmat(rows, n)
m = zeros(numel(rows), n);
for i = 1:numel(rows)
    m(i, rows(i)) = 1.0;
end
end
