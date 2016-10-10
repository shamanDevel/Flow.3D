#include "ConfigFile.h"

#include <fstream>
#include <sstream>
#include <iomanip>


namespace
{
	const std::string g_whitespaceChars(" \t\r\n");
	const std::string g_commentStartChars(";#");

	bool IsWhitespace(char c) { return g_whitespaceChars.find(c) != std::string::npos; }
	bool ContainsWhitespace(const std::string& str) { return str.find_first_of(g_whitespaceChars) != std::string::npos; }
	bool IsCommentStart(char c) { return g_commentStartChars.find(c) != std::string::npos; }
}


ConfigEntry::ConfigEntry(const std::string& name)
	: m_name(name)
{
}

ConfigEntry::ConfigEntry(const std::string& name, const std::string& value)
	: m_name(name), m_values(1, value)
{
}

ConfigEntry::ConfigEntry(const std::string& name, const std::vector<std::string>& values)
	: m_name(name), m_values(values)
{
}

ConfigEntry::ConfigEntry(const std::string& name, bool value)
	: m_name(name)
{
	AddValue(value);
}

ConfigEntry::ConfigEntry(const std::string& name, int value)
	: m_name(name)
{
	AddValue(value);
}

ConfigEntry::ConfigEntry(const std::string& name, float value)
	: m_name(name)
{
	AddValue(value);
}

ConfigEntry::ConfigEntry(const std::string& name, const tum3D::Vec2i& values)
	: m_name(name)
{
	AddValues(values);
}

ConfigEntry::ConfigEntry(const std::string& name, const tum3D::Vec3i& values)
	: m_name(name)
{
	AddValues(values);
}

ConfigEntry::ConfigEntry(const std::string& name, const tum3D::Vec4i& values)
	: m_name(name)
{
	AddValues(values);
}

ConfigEntry::ConfigEntry(const std::string& name, const tum3D::Vec2f& values)
	: m_name(name)
{
	AddValues(values);
}

ConfigEntry::ConfigEntry(const std::string& name, const tum3D::Vec3f& values)
	: m_name(name)
{
	AddValues(values);
}

ConfigEntry::ConfigEntry(const std::string& name, const tum3D::Vec4f& values)
	: m_name(name)
{
	AddValues(values);
}

ConfigEntry::ConfigEntry(const std::string& name, const tum3D::Mat3f& values)
	: m_name(name)
{
	AddValues(values);
}

ConfigEntry::ConfigEntry(const std::string& name, const tum3D::Mat4f& values)
	: m_name(name)
{
	AddValues(values);
}


void ConfigEntry::AddValue(bool value)
{
	std::ostringstream str;
	str << value;
	AddValue(str.str());
}

void ConfigEntry::AddValue(int value)
{
	std::ostringstream str;
	str << value;
	AddValue(str.str());
}

void ConfigEntry::AddValue(float value)
{
	std::ostringstream str;
	str << std::fixed << value;
	AddValue(str.str());
}


void ConfigEntry::AddValues(const tum3D::Vec2i& values)
{
	AddValue(values.x());
	AddValue(values.y());
}

void ConfigEntry::AddValues(const tum3D::Vec3i& values)
{
	AddValue(values.x());
	AddValue(values.y());
	AddValue(values.z());
}

void ConfigEntry::AddValues(const tum3D::Vec4i& values)
{
	AddValue(values.x());
	AddValue(values.y());
	AddValue(values.z());
	AddValue(values.w());
}

void ConfigEntry::AddValues(const tum3D::Vec2f& values)
{
	AddValue(values.x());
	AddValue(values.y());
}

void ConfigEntry::AddValues(const tum3D::Vec3f& values)
{
	AddValue(values.x());
	AddValue(values.y());
	AddValue(values.z());
}

void ConfigEntry::AddValues(const tum3D::Vec4f& values)
{
	AddValue(values.x());
	AddValue(values.y());
	AddValue(values.z());
	AddValue(values.w());
}

void ConfigEntry::AddValues(const tum3D::Mat3f& values)
{
	for(int i = 0; i < 9; i++) {
		AddValue(values[i]);
	}
}

void ConfigEntry::AddValues(const tum3D::Mat4f& values)
{
	for(int i = 0; i < 16; i++) {
		AddValue(values[i]);
	}
}


template<typename T>
static bool Parse(const std::string& str, T& value)
{
	std::istringstream stream(str);
	stream >> value;
	return !stream.fail();
}

template<>
static bool Parse<std::string>(const std::string& str, std::string& value)
{
	value = str;
	return true;
}

template<typename T>
bool ConfigEntry::GetValue(const std::string& func, T& value, size_t index) const
{
	if(index >= m_values.size()) {
		std::cerr << "ConfigEntry::" << func << ": Error: " << m_name.c_str() << " has less than " << (index+1) << " values" << std::endl;
		return false;
	}

	if(!Parse(m_values[index], value)) {
		std::cerr << "ConfigEntry::" << func << ": Error: Failed parsing " << m_name.c_str() << " value #" << index << " = \"" << m_values[index].c_str() << "\"" << std::endl;
		return false;
	}
	return true;
}

bool ConfigEntry::GetValueAsBool(bool& value, size_t index) const
{
	return GetValue("GetValueAsBool", value, index);
}

bool ConfigEntry::GetValueAsInt(int& value, size_t index) const
{
	return GetValue("GetValueAsInt", value, index);
}

bool ConfigEntry::GetValueAsFloat(float& value, size_t index) const
{
	return GetValue("GetValueAsFloat", value, index);
}


template<typename T>
bool ConfigEntry::Get(const std::string& func, T& value) const
{
	if(m_values.size() != 1) {
		std::cerr << "ConfigEntry::" << func << ": Error: " << m_name.c_str() << " has != 1 values" << std::endl;
		return false;
	}
	return GetValue(func, value);
}

template<int N, typename T>
bool ConfigEntry::GetVec(const std::string& func, Vec<N, T>& value) const
{
	if(m_values.size() != N) {
		std::cerr << "ConfigEntry::" << func << ": Error: " << m_name.c_str() << " has != " << N << " values" << std::endl;
		return false;
	}
	bool ok = true;
	for(int i = 0; i < N; i++) {
		ok = GetValue(func, value[i], i) && ok;
	}
	return ok;
}

template<int M, int N, typename T>
bool ConfigEntry::GetMat(const std::string& func, Mat<M, N, T>& value) const
{
	if(m_values.size() != M * N) {
		std::cerr << "ConfigEntry::" << func << ": Error: " << m_name.c_str() << " has != " << (M * N) << " values" << std::endl;
		return false;
	}
	bool ok = true;
	for(int i = 0; i < M * N; i++) {
		ok = GetValue(func, value[i], i) && ok;
	}
	return ok;
}

bool ConfigEntry::GetAsString(std::string& value) const
{
	return Get("GetAsString", value);
}

bool ConfigEntry::GetAsBool(bool& value) const
{
	return Get("GetAsBool", value);
}

bool ConfigEntry::GetAsInt(int& value) const
{
	return Get("GetAsInt", value);
}

bool ConfigEntry::GetAsFloat(float& value) const
{
	return Get("GetAsFloat", value);
}

bool ConfigEntry::GetAsVec2i(tum3D::Vec2i& value) const
{
	return GetVec("GetAsVec2i", value);
}

bool ConfigEntry::GetAsVec3i(tum3D::Vec3i& value) const
{
	return GetVec("GetAsVec3i", value);
}

bool ConfigEntry::GetAsVec4i(tum3D::Vec4i& value) const
{
	return GetVec("GetAsVec4i", value);
}

bool ConfigEntry::GetAsVec2f(tum3D::Vec2f& value) const
{
	return GetVec("GetAsVec2f", value);
}

bool ConfigEntry::GetAsVec3f(tum3D::Vec3f& value) const
{
	return GetVec("GetAsVec3f", value);
}

bool ConfigEntry::GetAsVec4f(tum3D::Vec4f& value) const
{
	return GetVec("GetAsVec4f", value);
}

bool ConfigEntry::GetAsMat3f(tum3D::Mat3f& value) const
{
	return GetMat("GetAsMat3f", value);
}

bool ConfigEntry::GetAsMat4f(tum3D::Mat4f& value) const
{
	return GetMat("GetAsMat4f", value);
}


std::ostream& operator<<(std::ostream& o, const ConfigEntry& entry)
{
	o << entry.m_name;
	for(size_t i = 0; i < entry.m_values.size(); i++) {
		if(ContainsWhitespace(entry.m_values[i])) {
			o << " \"" << entry.m_values[i] << "\"";
		} else {
			o << " " << entry.m_values[i];
		}
	}
	o << "\n";

	return o;
}


ConfigSection::ConfigSection(const std::string& name)
	: m_name(name)
{
}

std::ostream& operator<<(std::ostream& o, const ConfigSection& section)
{
	o << "[" << section.m_name << "]\n";
	for(size_t i = 0; i < section.m_entries.size(); i++) {
		o << "\t" << section.m_entries[i];
	}
	o << "\n";

	return o;
}


bool ConfigFile::Read(const std::string& filename)
{
	m_sections.clear();

	std::ifstream file(filename);
	if(!file.good()) {
		std::cerr << "ConfigFile::Read: Error opening file " << filename << " for reading" << std::endl;
		return false;
	}

	int curLine = 0;
	std::string line;
	while(!file.eof()) {
		// get next line
		++curLine;
		std::getline(file, line);
		if(file.bad()) {
			std::cerr << "ConfigFile::Read: Error reading from file " << filename << " in line " << curLine << std::endl;
			return false;
		}

		// this seems to happen when the last line of a file is empty..?!
		if(file.fail()) {
			continue;
		}

		size_t lineStart = line.find_first_not_of(g_whitespaceChars);
		size_t lineEnd   = line.find_first_of(g_commentStartChars);

		if(lineStart == std::string::npos) {
			// empty line
			continue;
		}
		if(IsCommentStart(line[lineStart])) {
			// comment
			continue;
		}

		if(line[lineStart] == '[') {
			// section start
			size_t start = lineStart + 1;
			size_t end = line.find(']', start);

			if(end >= lineEnd) {
				// missing closing bracket
				std::cerr << filename << "(" << curLine << "): Parse error: missing ']'" << std::endl;
				return false;
			}

			// make sure rest of line is empty or contains only a comment
			size_t rest = line.find_first_not_of(g_whitespaceChars, end + 1);
			if(rest < lineEnd) {
				std::cerr << filename << "(" << curLine << "): Parse error: illegal content after ']'" << std::endl;
				return false;
			}

			// add section
			std::string sectionName = line.substr(start, end - start);
			m_sections.push_back(ConfigSection(sectionName));
		} else {
			// value entry

			// first make sure that we already have a section
			if(m_sections.empty()) {
				std::cerr << filename << "(" << curLine << "): Parse error: entry before first section" << std::endl;
				return false;
			}

			// get name
			size_t start = lineStart;
			size_t end = line.find_first_of(g_whitespaceChars, start);
			std::string entryName = line.substr(start, end - start);

			// get values
			std::vector<std::string> entryValues;

			// find first value
			start = (end < std::string::npos ? line.find_first_not_of(g_whitespaceChars, end + 1) : end);
			while(start < lineEnd) {
				// find end of value
				bool quoted = false;
				if(start < lineEnd && line[start] == '\"') {
					quoted = true;
					// quoted entry, find end quote
					end = line.find('\"', start + 1);
					if(end >= lineEnd) {
						std::cerr << filename << "(" << curLine << "): Parse error: missing closing quote" << std::endl;
						return false;
					}
					// don't extract quotes
					++start;
				} else {
					end = std::min(lineEnd, line.find_first_of(g_whitespaceChars, start));
				}

				// extract value
				std::string value = line.substr(start, end - start);
				entryValues.push_back(value);

				if(quoted) ++end;

				// find start of next value
				start = line.find_first_not_of(g_whitespaceChars, end);
			}

			// add entry
			m_sections.back().GetEntries().push_back(ConfigEntry(entryName, entryValues));
		}
	}
	
	return true;
}

bool ConfigFile::Write(const std::string& filename) const
{
	std::ofstream file(filename);
	if(!file.good()) {
		std::cerr << "Error opening file " << filename << " for writing" << std::endl;
		return false;
	}

	for(size_t i = 0; i < m_sections.size(); i++) {
		file << m_sections[i];
	}

	return true;
}
