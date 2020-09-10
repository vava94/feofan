/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "cudaFont.h"
#include "utils.hpp"

#include <cuda_runtime.h>

#if !_HAS_CXX17
#ifdef _MSC_VER
#ifndef _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#endif
#endif
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb/stb_truetype.h"

#define TAG std::string("FONT: ")

//#define DEBUG_FONT


// Struct for one character to render
struct __align__(16) GlyphCommand
{
	short x;		// x coordinate origin in output image to begin drawing the glyph at 
	short y;		// y coordinate origin in output image to begin drawing the glyph at 
	short u;		// x texture coordinate in the baked font map where the glyph resides
	short v;		// y texture coordinate in the baked font map where the glyph resides 
	short width;	// width of the glyph in pixels
	short height;	// height of the glyph in pixels
};

template<typename T>
__global__ void gpuRectFill( T* input, T* output, int width, int height,
                             float4* rects, int numRects, float4 color )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= width || y >= height )
        return;

    T px = input[ y * width + x ];

    const float fx = x;
    const float fy = y;

    const float alpha = color.w / 255.0f;
    const float ialph = 1.0f - alpha;

    for( int nr=0; nr < numRects; nr++ )
    {
        const float4 r = rects[nr];

        if( fy >= r.y && fy <= r.w && fx >= r.x && fx <= r.z )
        {
            px.x = alpha * color.x + ialph * px.x;
            px.y = alpha * color.y + ialph * px.y;
            px.z = alpha * color.z + ialph * px.z;
        }
    }

    output[y * width + x] = px;
}

template<typename T>
__global__ void gpuRectFillBox( T* input, T* output, int imgWidth, int imgHeight, int x0, int y0, int boxWidth, int boxHeight, const float4 color )
{
    const int box_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int box_y = blockIdx.y * blockDim.y + threadIdx.y;

    if( box_x >= boxWidth || box_y >= boxHeight )
        return;

    const int x = box_x + x0;
    const int y = box_y + y0;

    if( x >= imgWidth || y >= imgHeight )
        return;

    T px = input[ y * imgWidth + x ];

    const float alpha = color.w / 255.0f;
    const float ialph = 1.0f - alpha;

    px.x = alpha * color.x + ialph * px.x;
    px.y = alpha * color.y + ialph * px.y;
    px.z = alpha * color.z + ialph * px.z;

    output[y * imgWidth + x] = px;
}

template<typename T>
cudaError_t launchRectFill( T* input, T* output, size_t width, size_t height, float4* rects, int numRects, const float4& color )
{
    if( !input || !output || width == 0 || height == 0 || !rects || numRects == 0 )
        return cudaErrorInvalidValue;

    // if input and output are the same image, then we can use the faster method
    // which draws 1 box per kernel, but doesn't copy pixels that aren't inside boxes
    if( input == output )
    {
        for( int n=0; n < numRects; n++ )
        {
            const int boxWidth = (int)(rects[n].z - rects[n].x);
            const int boxHeight = (int)(rects[n].w - rects[n].y);

            // launch kernel
            const dim3 blockDim(8, 8);
            const dim3 gridDim(iDivUp(boxWidth,blockDim.x), iDivUp(boxHeight,blockDim.y));

            gpuRectFillBox<T><<<gridDim, blockDim>>>(input, output, width, height, (int)rects[n].x, (int)rects[n].y, boxWidth, boxHeight, color);
        }
    }
    else
    {
        // launch kernel
        const dim3 blockDim(8, 8);
        const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

        gpuRectFill<T><<<gridDim, blockDim>>>(input, output, width, height, rects, numRects, color);
    }

    return cudaGetLastError();
}

cudaError_t cudaRectFill( void* input, void* output, size_t width, size_t height, float4* rects, int numRects, const float4& color )
{
    if( !input || !output || width == 0 || height == 0 || !rects || numRects == 0 )
        return cudaErrorInvalidValue;

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

    return launchRectFill<uchar3>((uchar3*)input, (uchar3*)output, width, height, rects, numRects, color);

    return cudaErrorInvalidValue;
}


// adaptFontSize
float adaptFontSize( uint32_t dimension )
{
	const float max_font = 32.0f;
	const float min_font = 28.0f;

	const uint32_t max_dim = 1536;
	const uint32_t min_dim = 768;

	if( dimension > max_dim )
		dimension = max_dim;

	if( dimension < min_dim )
		dimension = min_dim;

	const float dim_ratio = float(dimension - min_dim) / float(max_dim - min_dim);
	return min_font + dim_ratio * (max_font - min_font);

}


// constructor
cudaFont::cudaFont()
{
	mCommandCPU = NULL;
	mCommandGPU = NULL;
	mCmdIndex   = 0;

	mFontMapCPU = NULL;
	mFontMapGPU = NULL;

	mRectsCPU   = NULL;
	mRectsGPU   = NULL;
	mRectIndex  = 0;

	mFontMapWidth  = 256;
	mFontMapHeight = 256;
}



// destructor
cudaFont::~cudaFont()
{
	if( mRectsCPU != NULL )
	{
		cudaFreeHost(mRectsCPU);
		
		mRectsCPU = NULL; 
		mRectsGPU = NULL;
	}

	if( mCommandCPU != NULL )
	{
		cudaFreeHost(mCommandCPU);
		
		mCommandCPU = NULL; 
		mCommandGPU = NULL;
	}

	if( mFontMapCPU != NULL )
	{
		cudaFreeHost(mFontMapCPU);
		
		mFontMapCPU = NULL; 
		mFontMapGPU = NULL;
	}
}


// Create
cudaFont* cudaFont::Create( float size )
{
	// default fonts	
	std::vector<std::string> fonts;
	
	fonts.emplace_back("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf");
	fonts.emplace_back("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");

	return Create(fonts, size);
}


// Create
cudaFont* cudaFont::Create( const std::vector<std::string>& fonts, float size )
{
	const uint32_t numFonts = fonts.size();

	for( uint32_t n=0; n < numFonts; n++ )
	{
		cudaFont* font = Create(fonts[n].c_str(), size);

		if( font != NULL )
			return font;
	}

	return NULL;
}


// Create
cudaFont* cudaFont::Create( const char* font, float size )
{
	// verify parameters
	if( !font )
		return Create(size);

	// create new font
	auto c = new cudaFont();
		
	if( !c->init(font, size) )
	{
		delete c;
		return nullptr;
	}
	return c;
}


// init
bool cudaFont::init( const char* filename, float size )
{
	// validate parameters
	if( !filename )
		return false;

	// verify that the font file exists and get its size
	const size_t ttf_size = std::experimental::filesystem::file_size(filename);

	if( !ttf_size )
	{
		log(TAG + "font doesn't exist or empty file " + std::string (filename), 2);
 		return false;
	}

	// allocate memory to store the font file
	void* ttf_buffer = malloc(ttf_size);

	if( !ttf_buffer ){
        log(TAG + "failed to allocate" + std::to_string(ttf_size) + "byte buffer for reading " + std::string (filename), 2);
		return false;
	}

	// open the font file
	FILE* ttf_file = fopen(filename, "rb");

	if( !ttf_file )	{
        log(TAG + "failed to open " + std::string (filename) + "for reading", 2);
		free(ttf_buffer);
		return false;
	}

	// read the font file
	const size_t ttf_read = fread(ttf_buffer, 1, ttf_size, ttf_file);

	fclose(ttf_file);

	if( ttf_read != ttf_size )
	{
	    log(TAG + "failed to read contents of " + std::string(filename),2);
		free(ttf_buffer);
		return false;
	}

	// buffer that stores the coordinates of the baked glyphs
	stbtt_bakedchar bakeCoords[NumGlyphs];

	// increase the size of the bitmap until all the glyphs fit
	while(true)
	{
		// allocate memory for the packed font texture (alpha only)
		const size_t fontMapSize = mFontMapWidth * mFontMapHeight * sizeof(unsigned char);

		if (!utils::cudaAllocMapped((void**)&mFontMapCPU, (void**)&mFontMapGPU, fontMapSize) )
		{
            log(TAG + "failed to allocate " + std::to_string(fontMapSize) + " bytes to store " +
            std::to_string(mFontMapWidth) + "x" + std::to_string(mFontMapHeight) + "font map.", 2);
			free(ttf_buffer);
			return false;
		}

		// attempt to pack the bitmap
		const int result = stbtt_BakeFontBitmap((uint8_t*)ttf_buffer, 0, size, 
										mFontMapCPU, mFontMapWidth, mFontMapHeight,
									     FirstGlyph, NumGlyphs, bakeCoords);

		if( result == 0 )
		{
		    log(TAG + "failed to bake font bitmap " + filename, 2);
			free(ttf_buffer);
			return false;
		}
		else if( result < 0 )
		{
			const int glyphsPacked = -result;

			if( glyphsPacked == NumGlyphs )
			{
			    log(TAG + "packed " + std::to_string(NumGlyphs) + " glyphs in " + std::to_string(mFontMapWidth) +
			    "x" + std::to_string(mFontMapHeight), 2);
				break;
			}

		#ifdef DEBUG_FONT
			LogDebug(LOG_CUDA "fit only %i of %u font glyphs in %ux%u bitmap\n", glyphsPacked, NumGlyphs, mFontMapWidth, mFontMapHeight);
		#endif

			cudaFreeHost(mFontMapCPU);
		
			mFontMapCPU = NULL; 
			mFontMapGPU = NULL;

			mFontMapWidth *= 2;
			mFontMapHeight *= 2;

		#ifdef DEBUG_FONT
			LogDebug(LOG_CUDA "attempting to pack font with %ux%u bitmap...\n", mFontMapWidth, mFontMapHeight);
		#endif
			continue;
		}
		else
		{
		#ifdef DEBUG_FONT
			LogDebug(LOG_CUDA "packed %u glyphs in %ux%u bitmap (font size=%.0fpx)\n", NumGlyphs, mFontMapWidth, mFontMapHeight, size);
		#endif		
			break;
		}
	}

	// free the TTF font data
	free(ttf_buffer);

	// store texture baking coordinates
	for( uint32_t n=0; n < NumGlyphs; n++ )
	{
		mGlyphInfo[n].x = bakeCoords[n].x0;
		mGlyphInfo[n].y = bakeCoords[n].y0;

		mGlyphInfo[n].width  = bakeCoords[n].x1 - bakeCoords[n].x0;
		mGlyphInfo[n].height = bakeCoords[n].y1 - bakeCoords[n].y0;

		mGlyphInfo[n].xAdvance = bakeCoords[n].xadvance;
		mGlyphInfo[n].xOffset  = bakeCoords[n].xoff;
		mGlyphInfo[n].yOffset  = bakeCoords[n].yoff;

	#ifdef DEBUG_FONT
		// debug info
		const char c = n + FirstGlyph;
		LogDebug("Glyph %u: '%c' width=%hu height=%hu xOffset=%.0f yOffset=%.0f xAdvance=%0.1f\n", n, c, mGlyphInfo[n].width, mGlyphInfo[n].height, mGlyphInfo[n].xOffset, mGlyphInfo[n].yOffset, mGlyphInfo[n].xAdvance);
	#endif	
	}

	// allocate memory for GPU command buffer	
	if (!utils::cudaAllocMapped(&mCommandCPU, &mCommandGPU, sizeof(GlyphCommand) * MaxCommands) )
		return false;
	
	// allocate memory for background rect buffers
	if (!utils::cudaAllocMapped((void**)&mRectsCPU, (void**)&mRectsGPU, sizeof(float4) * MaxCommands) )
		return false;

	return true;
}


/*inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}*/

inline __host__ __device__ float4 alpha_blend( const float4& bg, const float4& fg )
{
	const float alpha = fg.w / 255.0f;
	const float ialph = 1.0f - alpha;
	
	return make_float4(alpha * fg.x + ialph * bg.x,
				    alpha * fg.y + ialph * bg.y,
				    alpha * fg.z + ialph * bg.z,
				    bg.w);
} 


template<typename T>
__global__ void gpuOverlayText( unsigned char* font, int fontWidth, GlyphCommand* commands,
						  T* input, T* output, int imgWidth, int imgHeight, float4 color ) 
{
	const GlyphCommand cmd = commands[blockIdx.x];

	if( threadIdx.x >= cmd.width || threadIdx.y >= cmd.height )
		return;

	const int x = cmd.x + threadIdx.x;
	const int y = cmd.y + threadIdx.y;

	if( x < 0 || y < 0 || x >= imgWidth || y >= imgHeight )
		return;

	const int u = cmd.u + threadIdx.x;
	const int v = cmd.v + threadIdx.y;

	const float px_glyph = font[v * fontWidth + u];

	const float4 px_font = make_float4(px_glyph * color.x, px_glyph * color.y, px_glyph * color.z, px_glyph * color.w);
	const float4 px_in   = cast_vec<float4>(input[y * imgWidth + x]);

	output[y * imgWidth + x] = cast_vec<T>(alpha_blend(px_in, px_font));	 
}


// cudaOverlayText
cudaError_t cudaOverlayText( unsigned char* font, const int2& maxGlyphSize, size_t fontMapWidth,
					    GlyphCommand* commands, size_t numCommands, const float4& fontColor, 
					    void* input, void* output, size_t imgWidth, size_t imgHeight)
{
	if( !font || !commands || !input || !output || numCommands == 0 || fontMapWidth == 0 || imgWidth == 0 || imgHeight == 0 )
		return cudaErrorInvalidValue;

	const float4 color_scaled = make_float4( fontColor.x / 255.0f, fontColor.y / 255.0f, fontColor.z / 255.0f, fontColor.w / 255.0f );
	
	// setup arguments
	const dim3 block(maxGlyphSize.x, maxGlyphSize.y);
	const dim3 grid(numCommands);

    gpuOverlayText<uchar3><<<grid, block>>>(font, fontMapWidth, commands, (uchar3*)input, (uchar3*)output, imgWidth, imgHeight, color_scaled);

	return cudaGetLastError();
}


// Overlay
bool cudaFont::OverlayText( void* image, uint32_t width, uint32_t height,
					   const std::vector< std::pair< std::string, int2 > >& strings, 
					   const float4& color, const float4& bg_color, int bg_padding )
{
	const uint32_t numStrings = strings.size();

	if( !image || width == 0 || height == 0 || numStrings == 0 )
		return false;
	
	const bool has_bg = bg_color.w > 0.0f;
	int2 maxGlyphSize = make_int2(0,0);

	int numCommands = 0;
	int numRects = 0;
	int maxChars = 0;

	// find the bg rects and total char count
	for( uint32_t s=0; s < numStrings; s++ )
		maxChars += strings[s].first.size();

	// reset the buffer indices if we need the space
	if( mCmdIndex + maxChars >= MaxCommands )
		mCmdIndex = 0;

	if( has_bg && mRectIndex + numStrings >= MaxCommands )
		mRectIndex = 0;

	// generate glyph commands and bg rects
	for( uint32_t s=0; s < numStrings; s++ )
	{
		const uint32_t numChars = strings[s].first.size();
		
		if( numChars == 0 )
			continue;

		// determine the max 'height' of the string
		int maxHeight = 0;

		for( uint32_t n=0; n < numChars; n++ )
		{
			char c = strings[s].first[n];
			
			if( c < FirstGlyph || c > LastGlyph )
				continue;
			
			c -= FirstGlyph;

			const int yOffset = abs((int)mGlyphInfo[c].yOffset);

			if( maxHeight < yOffset )
				maxHeight = yOffset;
		}

	#ifdef DEBUG_FONT
		LogDebug(LOG_CUDA "max glyph height:  %i\n", maxHeight);
	#endif

		// get the starting position of the string
		int2 pos = strings[s].second;

		if( pos.x < 0 )
			pos.x = 0;

		if( pos.y < 0 )
			pos.y = 0;
		
		pos.y += maxHeight;

		// reset the background rect if needed
		if( has_bg )
			mRectsCPU[mRectIndex] = make_float4(width, height, 0, 0);

		// make a glyph command for each character
		for( uint32_t n=0; n < numChars; n++ )
		{
			char c = strings[s].first[n];
			
			// make sure the character is in range
			if( c < FirstGlyph || c > LastGlyph )
				continue;
			
			c -= FirstGlyph;	// rebase char against glyph 0
			
			// fill the next command
			GlyphCommand* cmd = ((GlyphCommand*)mCommandCPU) + mCmdIndex + numCommands;

			cmd->x = pos.x;
			cmd->y = pos.y + mGlyphInfo[c].yOffset;
			cmd->u = mGlyphInfo[c].x;
			cmd->v = mGlyphInfo[c].y;

			cmd->width  = mGlyphInfo[c].width;
			cmd->height = mGlyphInfo[c].height;
		
			// advance the text position
			pos.x += mGlyphInfo[c].xAdvance;

			// track the maximum glyph size
			if( maxGlyphSize.x < mGlyphInfo[n].width )
				maxGlyphSize.x = mGlyphInfo[n].width;

			if( maxGlyphSize.y < mGlyphInfo[n].height )
				maxGlyphSize.y = mGlyphInfo[n].height;

			// expand the background rect
			if( has_bg )
			{
				float4* rect = mRectsCPU + mRectIndex + numRects;

				if( cmd->x < rect->x )
					rect->x = cmd->x;

				if( cmd->y < rect->y )
					rect->y = cmd->y;

				const float x2 = cmd->x + cmd->width;
				const float y2 = cmd->y + cmd->height;

				if( x2 > rect->z )
					rect->z = x2;

				if( y2 > rect->w )
					rect->w = y2;
			}

			numCommands++;
		}

		if( has_bg )
		{
			float4* rect = mRectsCPU + mRectIndex + numRects;

			// apply padding
			rect->x -= bg_padding;
			if (rect->x < 0) rect->x = 0;
			rect->y -= bg_padding;
            if (rect->y < 0) rect->y = 0;
			rect->z += bg_padding;
			rect->w += bg_padding;

			numRects++;
		}
	}

#ifdef DEBUG_FONT
	LogDebug(LOG_CUDA "max glyph size is %ix%i\n", maxGlyphSize.x, maxGlyphSize.y);
#endif

    if( has_bg && numRects > 0 )
        cudaRectFill(image, image, width, height, mRectsGPU + mRectIndex, numRects, bg_color);

	// draw text characters
	cudaOverlayText( mFontMapGPU, maxGlyphSize, mFontMapWidth,
				       ((GlyphCommand*)mCommandGPU) + mCmdIndex, numCommands, 
					  color, image, image, width, height);
			
	// advance the buffer indices
	mCmdIndex += numCommands;
	mRectIndex += numRects;
		   
	return true;
}


// Overlay
bool cudaFont::OverlayText( void* image, uint32_t width, uint32_t height,
					   const char* str, int x, int y, 
					   const float4& color, const float4& bg_color, int bg_padding )
{

	if( !str )
		return NULL;
		
	std::vector< std::pair< std::string, int2 > > list;
	
	list.push_back( std::pair< std::string, int2 >( str, make_int2(x,y) ));

	return OverlayText(image, width, height, list, color, bg_color, bg_padding);
}


// TextExtents
int4 cudaFont::TextExtents( const char* str, int x, int y )
{
	if( !str )
		return make_int4(0,0,0,0);

	const size_t numChars = strlen(str);

	// determine the max 'height' of the string
	int maxHeight = 0;

	for( uint32_t n=0; n < numChars; n++ )
	{
		char c = str[n];
		
		if( c < FirstGlyph || c > LastGlyph )
			continue;
		
		c -= FirstGlyph;

		const int yOffset = abs((int)mGlyphInfo[c].yOffset);

		if( maxHeight < yOffset )
			maxHeight = yOffset;
	}

	// get the starting position of the string
	int2 pos = make_int2(x,y);

	if( pos.x < 0 )
		pos.x = 0;

	if( pos.y < 0 )
		pos.y = 0;
	
	pos.y += maxHeight;


	// find the extents of the string
	for( uint32_t n=0; n < numChars; n++ )
	{
		char c = str[n];
		
		// make sure the character is in range
		if( c < FirstGlyph || c > LastGlyph )
			continue;
		
		c -= FirstGlyph;	// rebase char against glyph 0
		
		// advance the text position
		pos.x += mGlyphInfo[c].xAdvance;
	}

	return make_int4(x, y, pos.x, pos.y);
}
uint cudaFont::height() {
    uint _max = 0;
    for (const auto &_g : mGlyphInfo) {
        if(_g.height > _max) _max = _g.height;
    }
    return _max;
}

				
	
