#ifndef SH_MATH_H
#define SH_MATH_H

// View dependent color with real spherical harmonics, degrees 1 to 3. Same
// basis, signs and constants as the 3DGS reference renderer. Degree 0 is the
// gaussian's own rgb, so only the 15 coefficients of degrees 1 to 3 are kept:
// SH_STRIDE floats per gaussian, coefficient k of channel c at
// i * SH_STRIDE + k * 3 + c. tools/check_sh.py is the numpy version of this
// file and checks the gradient against finite differences.
//
// color seen from camera center c = rgb + sum_k sh_k Y_k(d), d = normalize(mu - c)

static const int SH_COEFFS = 15;
static const int SH_STRIDE = 45;

static const float SH_C1 = 0.4886025119029199;

// coefficients in use at a degree: 3 at 1, 8 at 2, 15 at 3
int sh_active(int degree) {
    return degree >= 3 ? 15 : (degree == 2 ? 8 : (degree == 1 ? 3 : 0));
}

void sh_basis(float3 d, out float Y[SH_COEFFS]) {
    float x = d.x, y = d.y, z = d.z;
    float xx = x * x, yy = y * y, zz = z * z;
    Y[0] = -SH_C1 * y;
    Y[1] = SH_C1 * z;
    Y[2] = -SH_C1 * x;
    Y[3] = 1.0925484305920792 * x * y;
    Y[4] = -1.0925484305920792 * y * z;
    Y[5] = 0.31539156525252005 * (2 * zz - xx - yy);
    Y[6] = -1.0925484305920792 * x * z;
    Y[7] = 0.5462742152960396 * (xx - yy);
    Y[8] = -0.5900435899266435 * y * (3 * xx - yy);
    Y[9] = 2.890611442640554 * x * y * z;
    Y[10] = -0.4570457994644658 * y * (4 * zz - xx - yy);
    Y[11] = 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy);
    Y[12] = -0.4570457994644658 * x * (4 * zz - xx - yy);
    Y[13] = 1.445305721320277 * z * (xx - yy);
    Y[14] = -0.5900435899266435 * x * (xx - 3 * yy);
}

// dY_k/dd with x, y, z treated as free, the normalization is applied after
void sh_basis_grad(float3 d, out float3 G[SH_COEFFS]) {
    float x = d.x, y = d.y, z = d.z;
    float xx = x * x, yy = y * y, zz = z * z;
    G[0] = float3(0, -SH_C1, 0);
    G[1] = float3(0, 0, SH_C1);
    G[2] = float3(-SH_C1, 0, 0);
    G[3] = 1.0925484305920792 * float3(y, x, 0);
    G[4] = -1.0925484305920792 * float3(0, z, y);
    G[5] = 0.31539156525252005 * float3(-2 * x, -2 * y, 4 * z);
    G[6] = -1.0925484305920792 * float3(z, 0, x);
    G[7] = 0.5462742152960396 * float3(2 * x, -2 * y, 0);
    G[8] = -0.5900435899266435 * float3(6 * x * y, 3 * xx - 3 * yy, 0);
    G[9] = 2.890611442640554 * float3(y * z, x * z, x * y);
    G[10] = -0.4570457994644658 * float3(-2 * x * y, 4 * zz - xx - 3 * yy, 8 * y * z);
    G[11] = 0.3731763325901154 * float3(-6 * x * z, -6 * y * z, 6 * zz - 3 * xx - 3 * yy);
    G[12] = -0.4570457994644658 * float3(4 * zz - 3 * xx - yy, -2 * x * y, 8 * x * z);
    G[13] = 1.445305721320277 * float3(2 * x * z, -2 * y * z, xx - yy);
    G[14] = -0.5900435899266435 * float3(3 * xx - 3 * yy, -6 * x * y, 0);
}

#endif // SH_MATH_H
