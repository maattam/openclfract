//
//  Author  : Matti Määttä
//  Summary :
//

#include "openclpbo.h"

#include <QDebug>
#include <QElapsedTimer>

#include <sstream>
#include <fstream>
#include <iterator>
#include <iostream>
#include <stdexcept>

OpenCLPBO::OpenCLPBO()
    : textureId_(0),
    image_width_(0), image_height_(0), cl_device_(-1),
    max_iterations_(500), double_precision_(false), cl_buffer_(0)
{
    memset(viewd_, 0, 4 * sizeof(cl_double));
    memset(viewf_, 0, 4 * sizeof(cl_float));
}

OpenCLPBO::~OpenCLPBO()
{
    deleteTexture();
}

void OpenCLPBO::init(const std::string& file, const std::string& kernel)
{
    cl_device_ = -1;

    cl_int err;
    cl::vector<cl::Platform> platforms;

    err = cl::Platform::get(&platforms);
    check_error(platforms.size() != 0 ? CL_SUCCESS : err, "No platforms found");

    // Selected platform
    cl::Platform platform;

    // Find the first GPU device
    for(int i = 0; i < platforms.size(); ++i)
    {
        err = platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices_);
        if(err == CL_SUCCESS)
        {
            platform = platforms[i];
            cl_device_ = 0;     // Select the primary device for now
            break;
        }
    }

    if(cl_device_ == -1)
    {
        check_error(-1, "No suitable devices found (CL_DEVICE_TYPE_GPU)");
    }

    cl_context_properties properties[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties) wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties) wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM, (cl_context_properties) platform(),
        0};

    context_ = cl::Context(devices_[cl_device_], properties, NULL, NULL, &err);
    check_error(err, "Create Context failed");

    // Check if double precision is supported
    std::string extensions = devices_[cl_device_].getInfo<CL_DEVICE_EXTENSIONS>();
    if(extensions.find("cl_khr_fp64") != std::string::npos)
    {
        double_precision_ = true;
    }

    // Load program
    loadProgram(file, kernel);

    // Build command queue
    queue_ =  cl::CommandQueue(context_, devices_[cl_device_], 0, &err);
    check_error(err, "CommandQueue::CommandQueue()");
}

void OpenCLPBO::resize(int width, int height)
{
    image_width_ = width;
    image_height_ = height;

    createTexture();
}

int OpenCLPBO::calculate()
{
    cl_int err;
    QElapsedTimer timer;
    cl::Event event;

    timer.start();
    check_error(cl_buffer_ != 0 ? CL_SUCCESS : -1, "Texture buffer not bound");
    check_error(cl_color_() != 0 ? CL_SUCCESS : -1, "Color buffer not bound");

    void* view_ptr = viewd_;
    size_t view_size = sizeof(cl_double) * 4;

    // Revert to single precision when needed
    if(!double_precision_)
    {
        view_ptr = viewf_;
        view_size = sizeof(cl_float) * 4;
    }

    cl::Buffer cl_view(context_, CL_MEM_READ_ONLY, view_size, NULL, &err);
    err = queue_.enqueueWriteBuffer(cl_view, CL_TRUE, 0, view_size, view_ptr, NULL, NULL);
    check_error(err, "Failed to allocate buffer");

    glFinish();
    err = clEnqueueAcquireGLObjects(queue_(), 1, &cl_buffer_, 0, NULL, NULL);
    check_error(err, "AcquireGLObjects failed");
    
    // Run program
    kernel_.setArg(0, cl_buffer_);
    kernel_.setArg(1, image_width_);
    kernel_.setArg(2, image_height_);
    kernel_.setArg(3, cl_view);
    kernel_.setArg(4, max_iterations_);
    kernel_.setArg(5, cl_color_);

    int workw = image_width_ + pool_size - (image_width_ % pool_size);
    int workh = image_height_ + pool_size - (image_height_ % pool_size);

    err = queue_.enqueueNDRangeKernel(kernel_, cl::NullRange,
        cl::NDRange(workw, workh),
        cl::NDRange(pool_size, pool_size),
        NULL, &event);

    check_error(err, "enqueueNDRangeKernel failed");
    event.wait();

    err = clEnqueueReleaseGLObjects(queue_(), 1, &cl_buffer_, 0, NULL, NULL);
    check_error(err, "ReleaseGLObjects failed");

    queue_.finish();
    return timer.elapsed();
}

void OpenCLPBO::createTexture()
{
    deleteTexture();

    // Generate texture id
    glGenTextures(1, &textureId_);

    // Set current texture
    glBindTexture(GL_TEXTURE_2D, textureId_);

    // Allocate memory
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image_width_, image_height_, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    check_error(glGetError(), "Out of video memory");

    // Set filtering mode
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    cl_int err;
    cl_buffer_ = clCreateFromGLTexture2D(context_(), CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, textureId_, &err);
    glBindTexture(GL_TEXTURE_2D, 0);

    check_error(err, "clCreateFromGLTexture2D");
}

void OpenCLPBO::deleteTexture()
{
    if(cl_buffer_ != 0)
    {
        clReleaseMemObject(cl_buffer_);
        cl_buffer_ = 0;
    }

    glDeleteTextures(1, &textureId_);
}

void OpenCLPBO::check_error(cl_int err, const std::string& name) const
{
    if(err != CL_SUCCESS)
    {
        std::stringstream ss;
        ss << name << " ( " << err << " )";

        throw std::runtime_error(ss.str());
    }
}

void OpenCLPBO::loadProgram(const std::string& file, const std::string& kernelstr)
{
    cl_int err;
    std::ifstream fs(file);
    check_error(fs.is_open() ? CL_SUCCESS : -1, "Failed to open file: " + file);

    std::string prog(std::istreambuf_iterator<char>(fs), (std::istreambuf_iterator<char>()));

    if(double_precision_)
        prog = "#define USE_DOUBLE 1\r\n" + prog;

    cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length()+1));
    cl::Program program(context_, source);
    err = program.build(devices_, "");

    if(err == CL_BUILD_PROGRAM_FAILURE)
    {
        auto info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices_[cl_device_]);
        check_error(-1, "Build error:\n" + info);
    }

    check_error(err, "Failed to build program");

    cl::Kernel kernel(program, kernelstr.c_str(), &err);
    check_error(err, "Failed to load kernel: " + kernelstr);

    kernel_ = kernel;
}

void OpenCLPBO::setView(double minRe, double maxRe, double minIm, double maxIm)
{
    if(double_precision_)
    {
        viewd_[0] = minRe; viewd_[1] = maxRe;
        viewd_[2] = minIm; viewd_[3] = maxIm;
    }

    else
    {
        viewf_[0] = minRe; viewf_[1] = maxRe;
        viewf_[2] = minIm; viewf_[3] = maxIm;
    }
}

void OpenCLPBO::precomputeColor(Color color)
{
    color_ = color;
    const size_t SIZE = max_iterations_;
    cl_float4* table = new cl_float4[SIZE];

    for(int i = 0; i < max_iterations_; ++i)
    {
        if(color_ == Color::Poly)
        {
            getColorPoly(i, table[i]);
        }

        else if(color_ == Color::Trig)
        {
            getColorTrig(i, table[i]);
        }
    }

    cl_int err;
    
    // Write color map to device memory
    cl_color_ = cl::Buffer(context_, CL_MEM_READ_ONLY, SIZE * sizeof(cl_float4), NULL, &err);
    err = queue_.enqueueWriteBuffer(cl_color_, CL_TRUE, 0, SIZE * sizeof(cl_float4), table, NULL, NULL);
    queue_.finish();

    delete [] table;
    check_error(err, "Precompute Color Failure");
}

void OpenCLPBO::getColorPoly(size_t x, cl_float4& color)
{
    double t = (double) x / (double) max_iterations_;

    color.s[0] = 9*(1-t)*t*t*t;
    color.s[1] = 15*(1-t)*(1-t)*t*t;
    color.s[2] = 8.5*(1-t)*(1-t)*(1-t)*t;
    color.s[3] = 1;
}

void OpenCLPBO::getColorTrig(size_t x, cl_float4& color)
{
    double t = (double) x / (double) max_iterations_;

    color.s[0] = 8.5*(1-t)*(1-t)*(1-t)*t;
    color.s[1] = 15*(1-t)*(1-t)*t*t;
    color.s[2] = 9*(1-t)*t*t*t;
    color.s[3] = 1;
}

void OpenCLPBO::max_iterations(cl_uint val)
{
    if(val < 100)
    {
        return;
    }

    max_iterations_ = val;
    precomputeColor(color_);
}

cl_uint OpenCLPBO::max_iterations() const { return max_iterations_; }

std::string OpenCLPBO::precision() const
{
    if(double_precision_)
        return "Double";

    return "Single";
}

GLuint OpenCLPBO::getTextureId() const
{
    return textureId_;
}