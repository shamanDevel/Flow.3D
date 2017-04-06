#include <global.h>

#include "AdvectMode.h"
#include "BrickIndexGPU.h"
#include "BrickRequestsGPU.h"
#include "IntegrationParamsGPU.h"
#include "LineInfoGPU.h"
#include "TextureCPU.h"
#include "TextureFilterMode.h"
#include "TracingCommon.h"
#include "VolumeInfoGPU.h"

#include "AdvectDenseCPU.h"
#include "Coords.cuh"
#include "IntegratorCommonCPU.h"
#include "TextureFilterCPU.h"

extern VolumeInfoGPU g_volumeInfo;
extern BrickIndexGPU g_brickIndex;
extern BrickRequestsGPU g_brickRequests;
extern IntegrationParamsGPU g_integrationParams;
extern LineInfoGPU g_lineInfo;

extern TextureCPU<float4> g_volume;


template<eAdvectMode advectMode, eTextureFilterMode filterMode>
void integrateOneStreamLineDenseCPU(uint lineIndex)
{
	uint lineLength = g_lineInfo.pVertexCounts[lineIndex];

	if(lineLength >= g_lineInfo.lineLengthMax)
		return;

	LineVertex vertex;
	// get initial position from checkpoints array
	vertex.Position = g_lineInfo.pCheckpoints[lineIndex].Position;
	vertex.Time     = g_lineInfo.pCheckpoints[lineIndex].Time;

	if(vertex.Time >= g_integrationParams.timeMax || g_volumeInfo.isOutsideOfDomain(vertex.Position))
		return;


	// find brick we're in
	float3 brickBoxMin;
	float3 brickBoxMax;
	float3 world2texOffset;
	float3 world2texScale;
	if(!findBrick(vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale)) {
		return;
	}

	// get velocity at initial position
	vertex.Velocity = g_volumeInfo.velocityScale * sampleVolume<filterMode, float4, float3>(g_volume, w2t(vertex.Position));

	vertex.LineID = lineIndex;


	// this always points to the next vertex to be written out
	LineVertex* pVertices = g_lineInfo.pVertices + lineIndex * g_lineInfo.vertexStride + lineLength;

	if(lineLength == 0) {
		// new line - build normal: arbitrary vector perpendicular to velocity
		float3 tangent = normalize(vertex.Velocity);
		vertex.Normal = cross(tangent, make_float3(1.0f, 0.0f, 0.0f));
		if(length(vertex.Normal) < 0.01f) vertex.Normal = cross(tangent, make_float3(0.0f, 1.0f, 0.0f));
		vertex.Normal = normalize(vertex.Normal);

		// write out initial vertex
		*pVertices++ = vertex;
		++lineLength;
	} else {
		// existing line - get old normal
		vertex.Normal = g_lineInfo.pCheckpoints[lineIndex].Normal;
	}

	// get the last vertex that was written out
	float3 lastOutPos  = (pVertices - 1)->Position;
	float  lastOutTime = (pVertices - 1)->Time;

	float deltaTime = g_lineInfo.pCheckpoints[lineIndex].DeltaT;

	uint step = 0;
	uint stepsAccepted = 0;

	// dense output
	const uint coeffCount = advectDenseInfo<advectMode>::OutputCoeffCount;
	float3 outputCoeffs[coeffCount];

	bool stayedInAvailableBrick = true;
	while(step < g_integrationParams.stepCountMax &&
		  vertex.Time < g_integrationParams.timeMax &&
		  lineLength < g_lineInfo.lineLengthMax)
	{
		float deltaTimeBak = deltaTime;
		// limit deltaTime ..
		// .. so we don't integrate past timeMax
		deltaTime = min(deltaTime, g_integrationParams.timeMax - vertex.Time);
		// .. so we don't leave the current brick's safe region
		float distMax = g_integrationParams.brickSafeMarginWorld + distanceToBrickBorder(vertex.Position, brickBoxMin, brickBoxMax);
		deltaTime = min(deltaTime, distMax / g_integrationParams.velocityMaxWorld);
		// integrate
		float deltaTimeThisStep = deltaTime;
		bool stepAccepted = advectDense<advectMode, filterMode>(
			g_volume,
			vertex.Position, vertex.Time, vertex.Velocity,
			deltaTime,
			outputCoeffs,
			world2texOffset, world2texScale,
			g_volumeInfo.velocityScale);
		++step;
		if(stepAccepted) {
			++stepsAccepted;

			// if we artificially limited deltaTime earlier, reset it now
			// (if we didn't, the new deltaTime is larger than the backup anyway)
			deltaTime = fmax(deltaTime, deltaTimeBak);

			//vertex.Jacobian = getJacobian<filterMode>(g_volume, w2t(vertex.Position));

			float3 posDiff = vertex.Position - lastOutPos;
			float timeDiff = vertex.Time     - lastOutTime;
			float posDiffSqr = dot(posDiff, posDiff);
			if((posDiffSqr >= g_integrationParams.outputPosDiffSquared) || (timeDiff >= g_integrationParams.outputTimeDiff)) {
				// write out interpolated positions
				uint intervalCount = max(1, uint(sqrt(posDiffSqr / g_integrationParams.outputPosDiffSquared)));
				intervalCount = min(intervalCount, g_lineInfo.lineLengthMax - lineLength);
				// interval == 0 corresponds to the old position, interval == intervalCount to the new one
				LineVertex tmpVertex = vertex;
				for(uint interval = 1; interval < intervalCount; ++interval) {
					float3 tmp[coeffCount];
					// position:
					// copy coefficients
					for(uint i = 0; i < coeffCount; ++i) {
						tmp[i] = outputCoeffs[i];
					}
					// evaluate bezier segment using de Casteljau's scheme
					float t = float(interval) / float(intervalCount);
					for(uint l = 1; l < coeffCount; ++l) {
						for(uint i = coeffCount - 1; i >= l; --i) {
							tmp[i] = (1.0f - t) * tmp[i - 1] + t * tmp[i];
						}
					}
					tmpVertex.Position = tmp[coeffCount - 1];
					tmpVertex.Time = vertex.Time - (1.0f - t) * deltaTimeThisStep;
					// velocity:
					for(uint i = 0; i < coeffCount - 1; ++i) {
						tmp[i] = outputCoeffs[i+1] - outputCoeffs[i];
					}
					for(uint l = 1; l < coeffCount - 1; ++l) {
						for(uint i = coeffCount - 2; i >= l; --i) {
							tmp[i] = (1.0f - t) * tmp[i - 1] + t * tmp[i];
						}
					}
					tmpVertex.Velocity = float(coeffCount - 1) * tmp[coeffCount - 2] / deltaTimeThisStep;

					// re-orthogonalize normal wrt. tangent == velocity direction
					float3 binormal = cross(tmpVertex.Velocity, tmpVertex.Normal);
					tmpVertex.Normal = normalize(cross(binormal, tmpVertex.Velocity));
					// and write out the interpolated vertex
					*pVertices++ = tmpVertex;
					++lineLength;
				}


				// re-orthogonalize normal wrt. tangent == velocity direction
				float3 binormal = cross(vertex.Velocity, tmpVertex.Normal);
				vertex.Normal = normalize(cross(binormal, vertex.Velocity));
				// write out final step position
				*pVertices++ = vertex;
				++lineLength;

				lastOutPos  = vertex.Position;
				lastOutTime = vertex.Time;
			} else {
				// even if we don't output anything, we still need to
				// re-orthogonalize normal wrt. tangent == velocity direction
				float3 binormal = cross(vertex.Velocity, vertex.Normal);
				vertex.Normal = normalize(cross(binormal, vertex.Velocity));
			}

			// check if we left the current brick
			if(!isInBrick(vertex.Position, brickBoxMin, brickBoxMax)) {
				bool isOutOfDomain = g_volumeInfo.isOutsideOfDomain(vertex.Position);
				if(isOutOfDomain) {
					// write out final position
					*pVertices++ = vertex;
					++lineLength;

					lastOutPos  = vertex.Position;
					lastOutTime = vertex.Time;
				}
				if(isOutOfDomain || !findBrick(vertex.Position, brickBoxMin, brickBoxMax, world2texOffset, world2texScale)) {
					// new brick isn't available (or we went out of the domain) - get outta here
					// (if we're still inside the domain, the new brick has already been requested in findBrick!)
					stayedInAvailableBrick = false;
					break;
				} else {
					// semi-HACK: update velocity from new brick (can be different to previous one because of lossy compression)
					//            this avoids excessively small time steps at some brick boundaries
					vertex.Velocity = g_volumeInfo.velocityScale * sampleVolume<filterMode, float4, float3>(g_volume, w2t(vertex.Position));
				}
			}
		}
	}


	g_lineInfo.pVertexCounts[lineIndex] = lineLength;
	//assert(g_lineInfo.pVertexCounts[lineIndex] < lineLengthMax);

	// update checkpoint for next integration round
	g_lineInfo.pCheckpoints[lineIndex].Position = vertex.Position;
	g_lineInfo.pCheckpoints[lineIndex].Time     = vertex.Time;
	g_lineInfo.pCheckpoints[lineIndex].Normal   = vertex.Normal;
	g_lineInfo.pCheckpoints[lineIndex].DeltaT   = deltaTime;

	g_lineInfo.pCheckpoints[lineIndex].StepsAccepted += stepsAccepted;
	g_lineInfo.pCheckpoints[lineIndex].StepsTotal    += step;

	// if the line is still alive and in an available brick, request it again for next round
	if(vertex.Time < g_integrationParams.timeMax &&
	   lineLength < g_lineInfo.lineLengthMax &&
	   stayedInAvailableBrick)
	{
		// find out which brick we're in now
		uint3 brickIndex = g_volumeInfo.getBrickIndex(vertex.Position);
		uint brickLinearIndex = g_volumeInfo.getBrickLinearIndex(brickIndex);
		// request it to be loaded
		g_brickRequests.requestBrickCPU(brickLinearIndex);
	}
}

template<eAdvectMode advectMode, eTextureFilterMode filterMode>
void integrateStreamLinesDenseCPU(uint lineCount)
{
	#pragma omp parallel for
	for(int lineIndex = 0; lineIndex < int(lineCount); lineIndex++)
	{
		integrateOneStreamLineDenseCPU<advectMode, filterMode>(lineIndex);
	}
}


void integratorKernelStreamLinesDenseCPU(const LineInfo& lineInfo, eAdvectMode advectMode, eTextureFilterMode filterMode)
{
	if(advectMode != ADVECT_RK547M)
	{
		printf("integratorKernelStreamLinesDenseCPU: unsupported advect mode %s, using RK547M\n", GetAdvectModeName(advectMode).c_str());
	}
	switch(filterMode)
	{
		case TEXTURE_FILTER_LINEAR:
			integrateStreamLinesDenseCPU<ADVECT_RK547M, TEXTURE_FILTER_LINEAR>(lineInfo.lineCount);
			break;
		case TEXTURE_FILTER_CATROM:
			integrateStreamLinesDenseCPU<ADVECT_RK547M, TEXTURE_FILTER_CATROM>(lineInfo.lineCount);
			break;
		case TEXTURE_FILTER_CATROM_STAGGERED:
			integrateStreamLinesDenseCPU<ADVECT_RK547M, TEXTURE_FILTER_CATROM>(lineInfo.lineCount);
			break;
		case TEXTURE_FILTER_LAGRANGE4:
			integrateStreamLinesDenseCPU<ADVECT_RK547M, TEXTURE_FILTER_LAGRANGE4>(lineInfo.lineCount);
			break;
		case TEXTURE_FILTER_LAGRANGE6:
			integrateStreamLinesDenseCPU<ADVECT_RK547M, TEXTURE_FILTER_LAGRANGE6>(lineInfo.lineCount);
			break;
		case TEXTURE_FILTER_LAGRANGE8:
			integrateStreamLinesDenseCPU<ADVECT_RK547M, TEXTURE_FILTER_LAGRANGE8>(lineInfo.lineCount);
			break;
		default:
			printf("integratorKernelStreamLinesDenseCPU: unsupported filter mode %s\n", GetTextureFilterModeName(filterMode).c_str());
	}
}
