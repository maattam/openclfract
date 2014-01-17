//
//  Author  : Matti Määttä
//  Summary : OpenCL pixel buffer object for generating fractal texture.
//

#ifndef OPENCLPBO_H
#define OPENCLPBO_H

#include <qglbuffer.h>
#include <string>

#define __NO_STD_VECTOR
#include <CL/cl.hpp>

class OpenCLPBO
{
public:
    enum { pool_size = 16 };
    enum Color { Poly, Trig };

    OpenCLPBO();
    ~OpenCLPBO();

    void init(const std::string& file, const std::string& kernel);
    void resize(int width, int height);
    int calculate();
    void setView(double minRe, double maxRe, double minIm, double maxIm);
    void precomputeColor(Color color);

    void max_iterations(cl_uint val);

    cl_uint max_iterations() const;
    std::string precision() const;
    GLuint getTextureId() const;

private:
    cl_uint image_width_;
    cl_uint image_height_;
    cl_uint max_iterations_;
    Color color_;
    int cl_device_;
    bool double_precision_;

    cl::Buffer cl_color_;
    cl_double viewd_[4];
    cl_float viewf_[4];
    cl_mem cl_buffer_;
    GLuint textureId_;

    cl::Context context_;
    cl::vector<cl::Device> devices_;
    cl::Kernel kernel_;
    cl::CommandQueue queue_;

    void createTexture();
    void deleteTexture();
    void check_error(signed int err, const std::string& name) const;
    void loadProgram(const std::string& file, const std::string& kernel);

    void getColorPoly(size_t x, cl_float4& color);
    void getColorTrig(size_t x, cl_float4& color);
};

#endif // OPENCLPBO_H