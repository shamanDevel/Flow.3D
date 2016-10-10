
template < eMeasure M, eTextureFilterMode F, eMeasureComputeMode C, eColorMode CM> 
__device__ inline void ms3SivrStep(
	float4& sum,
	bool& bWasInsideCoarse,
	bool& bWasInsideMid,
	bool& bWasInsideFine,
	const float3& world2texOffset,
	float world2texScale,
	const float3& rayPosTx,
	const float3& pos,
	const float3& step,
	const float3& rayDir,
	float measureScale,
	const float3& isoValues,
	const float4& isoColor1,
	const float4& isoColor2,
	const float4& isoColor3)
{
	float fDist[3] = {1e10, 1e10, 1e10};
	float4 vColors[3] = {make_float4(0,0,0,0), make_float4(0,0,0,0), make_float4(0,0,0,0)};
		
	//Coarse scale
	bool bInsideCoarse = (isoValues.x < getMeasure<M,F,C>(g_texVolume3, w2t(pos), measureScale));
	if (bInsideCoarse != bWasInsideCoarse) 
	{
		float factor = (bInsideCoarse ? 1.0f : 0.0f);
		float3 pp	 = binarySearch<M,F,C>(g_texVolume3, w2t(pos - factor * step), w2t(pos - (1.0f-factor) * step), isoValues.x, measureScale);
		float3 grad  = getGradient<M,F,C>(g_texVolume3, pp) * (bInsideCoarse ? 1.0f : -1.0f);
		vColors[0]	 = shadeScaleInvariant( rayDir, grad, isoColor1 );
		fDist[0]	 = lengthSq(pp-rayPosTx);
	}
	bWasInsideCoarse = bInsideCoarse;

	//Mid scale
	bool bInsideMid = (isoValues.y < getMeasure<M,F,C>(g_texVolume2, w2t(pos), measureScale));
	if (bInsideMid != bWasInsideMid) 
	{
		float factor = (bInsideMid ? 1.0f : 0.0f);
		float3 pp	 = binarySearch<M,F,C>(g_texVolume2, w2t(pos - factor * step), w2t(pos - (1.0f-factor) * step), isoValues.y, measureScale);
		float3 grad  = getGradient<M,F,C>(g_texVolume2, pp) * (bInsideMid ? 1.0f : -1.0f);
		vColors[1]   = shadeScaleInvariant( rayDir, grad, isoColor2 );
		fDist[1]	 = lengthSq(pp-rayPosTx);
	}
	bWasInsideMid = bInsideMid;

	//Small scale
	bool bInsideFine = (isoValues.z < getMeasure<M,F,C>(g_texVolume1, w2t(pos), measureScale));
	if (bInsideFine != bWasInsideFine)
	{
		float factor	 = (bInsideFine ? 1.0f : 0.0f);
		float3 pp		 = binarySearch<M,F,C>(g_texVolume1, w2t(pos - factor * step), w2t(pos - (1.0f-factor) * step), isoValues.z, measureScale);
		float3 grad		 = getGradient<M,F,C>(g_texVolume1, pp) * (bInsideFine ? 1.0f : -1.0f);
		float4 surfColor = getIsoColor<F,C,CM>(g_texVolume1, g_texVolume2, pp, isoColor3);
		vColors[2]		 = shadeScaleInvariant( rayDir, grad, surfColor );
		fDist[2]		 = lengthSq(pp-rayPosTx);
	}
	bWasInsideFine = bInsideFine;

	int idxs[3] = {0,1,2};
	if(fDist[1] < fDist[0])		 { idxs[0] = 1; idxs[1] = 0; }
	if(fDist[2] < fDist[idxs[1]]){ if(fDist[2] < fDist[idxs[0]]) { idxs[2] = idxs[1]; idxs[1] = idxs[0]; idxs[0] = 2; }else{ idxs[2] = idxs[1]; idxs[1] = 2; } }

	sum += (1.0f - sum.w) * vColors[idxs[0]];
	sum += (1.0f - sum.w) * vColors[idxs[1]];
	sum += (1.0f - sum.w) * vColors[idxs[2]];
}
