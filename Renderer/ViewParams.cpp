#include "ViewParams.h"
#include "Vec.h"

#include <cstring>

using namespace tum3D;


ViewParams::ViewParams()
{
	Reset();

	//m_lookAt = tum3D::Vec3f(0.3f, -0.72f, 0.0f);
	//m_viewDistance = 1.3f;
}

void ViewParams::Reset()
{
	m_lookAt.clear();
	m_rotationQuat.set(1.0f, 0.0f, 0.0f, 0.0f);
	m_viewDistance = 5.0f;
}


void ViewParams::ApplyConfig(const ConfigFile& config)
{
	Reset();

	const std::vector<ConfigSection>& sections = config.GetSections();
	for(size_t s = 0; s < sections.size(); s++)
	{
		const ConfigSection& section = sections[s];
		if(section.GetName() == "ViewParams")
		{
			// this is our section - parse entries
			for(size_t e = 0; e < section.GetEntries().size(); e++) {
				const ConfigEntry& entry = section.GetEntries()[e];

				std::string entryName = entry.GetName();

				if(entryName == "RotationQuat")
				{
					entry.GetAsVec4f(m_rotationQuat);
				}
				else if(entryName == "LookAt")
				{
					entry.GetAsVec3f(m_lookAt);
				}
				else if(entryName == "ViewDistance")
				{
					entry.GetAsFloat(m_viewDistance);
				}
				else
				{
					printf("WARNING: ViewParams::ApplyConfig: unknown entry \"%s\" ignored\n", entryName.c_str());
				}
			}
		}
	}
}

void ViewParams::WriteConfig(ConfigFile& config) const
{
	ConfigSection section("ViewParams");

	section.AddEntry(ConfigEntry("RotationQuat", m_rotationQuat));
	section.AddEntry(ConfigEntry("LookAt", m_lookAt));
	section.AddEntry(ConfigEntry("ViewDistance", m_viewDistance));

	config.AddSection(section);
}

Vec3f ViewParams::GetRightVector() const
{
	Mat4f rotationMat;
	convertQuaternionToRotMat(m_rotationQuat, rotationMat);

	Vec3f v = rotationMat.getRow(0);

	v = normalize(v);

	return v;
}

Vec3f ViewParams::GetViewDir() const
{
	Mat4f rotationMat;
	convertQuaternionToRotMat(m_rotationQuat, rotationMat);

	Vec3f v = rotationMat.getRow(2);

	v = normalize(v);

	return v;
}



Vec3f ViewParams::GetCameraPosition() const
{
	//Mat4f rotationMat;
	//convertQuaternionToRotMat(m_rotationQuat, rotationMat);

	//Vec3f viewDir = -Vec3f(rotationMat.getRow(2));
	Vec3f viewDir = -GetViewDir();
	Vec3f eyePos = m_lookAt - viewDir * m_viewDistance;

	return eyePos;
}

Mat4f ViewParams::BuildViewMatrix(EStereoEye eye, float eyeDistance) const
{
	Mat4f rotationMat;
	convertQuaternionToRotMat(m_rotationQuat, rotationMat);

	Vec3f viewDir = -Vec3f(rotationMat.getRow(2));
	Vec3f eyePos = m_lookAt - viewDir * m_viewDistance;

	Mat4f translation;
	translationMat(-eyePos, translation);

	Mat4f view = rotationMat * translation;

	float eyeOffset = 0.0f;
	switch(eye)
	{
		case EYE_LEFT:  eyeOffset =  0.5f * eyeDistance; break;
		case EYE_RIGHT: eyeOffset = -0.5f * eyeDistance; break;
		// EYE_CYCLOP: nop
	}
	Mat4f eyeOffsetMat; translationMat(eyeOffset, 0.0f, 0.0f, eyeOffsetMat);
	view = eyeOffsetMat * view;

	return view;
}

tum3D::Mat4f ViewParams::BuildRotationMatrix() const
{
	Mat4f rotationMat;
	convertQuaternionToRotMat(m_rotationQuat, rotationMat);
	return rotationMat;
}

bool ViewParams::operator==(const ViewParams& rhs) const
{
	return memcmp(this, &rhs, sizeof(ViewParams)) == 0;
}

bool ViewParams::operator!=(const ViewParams& rhs) const
{
	return !(*this == rhs);
}
