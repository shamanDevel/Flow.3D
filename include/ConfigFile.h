#ifndef __TUM3D__CONFIG_FILE_H__
#define __TUM3D__CONFIG_FILE_H__


#include <iostream>
#include <string>
#include <vector>

#include "Vec.h"


// A key-value(s) pair, in the form: key [value1 [value2 [...]]]
// The key may not contain spaces. The values may contain spaces, but no double quotes (TODO?).
class ConfigEntry
{
public:
	ConfigEntry(const std::string& name);
	ConfigEntry(const std::string& name, const std::string& value);
	ConfigEntry(const std::string& name, const std::vector<std::string>& values);
	ConfigEntry(const std::string& name, bool value);
	ConfigEntry(const std::string& name, int value);
	ConfigEntry(const std::string& name, float value);
	ConfigEntry(const std::string& name, const tum3D::Vec2i& values);
	ConfigEntry(const std::string& name, const tum3D::Vec3i& values);
	ConfigEntry(const std::string& name, const tum3D::Vec4i& values);
	ConfigEntry(const std::string& name, const tum3D::Vec2f& values);
	ConfigEntry(const std::string& name, const tum3D::Vec3f& values);
	ConfigEntry(const std::string& name, const tum3D::Vec4f& values);
	ConfigEntry(const std::string& name, const tum3D::Mat3f& values);
	ConfigEntry(const std::string& name, const tum3D::Mat4f& values);

	const std::string& GetName() const { return m_name; }

	// raw values access
	      std::vector<std::string>& GetValues()       { return m_values; }
	const std::vector<std::string>& GetValues() const { return m_values; }

	size_t GetValueCount() const { return m_values.size(); }

	// add values in various formats
	void AddValue(const std::string& value) { m_values.push_back(value); }
	void AddValue(bool value);
	void AddValue(int value);
	void AddValue(float value);

	void AddValues(const tum3D::Vec2i& values);
	void AddValues(const tum3D::Vec3i& values);
	void AddValues(const tum3D::Vec4i& values);
	void AddValues(const tum3D::Vec2f& values);
	void AddValues(const tum3D::Vec3f& values);
	void AddValues(const tum3D::Vec4f& values);
	void AddValues(const tum3D::Mat3f& values);
	void AddValues(const tum3D::Mat4f& values);

	// extract/parse individual values - return false if parsing fails, or index out of bounds
	bool GetValueAsBool(bool& value, size_t index = 0) const;
	bool GetValueAsInt(int& value, size_t index = 0) const;
	bool GetValueAsFloat(float& value, size_t index = 0) const;

	// parse whole entry - return false is parsing fails or if value count doesn't match
	bool GetAsString(std::string& value) const;
	bool GetAsBool(bool& value) const;
	bool GetAsInt(int& value) const;
	bool GetAsFloat(float& value) const;

	bool GetAsVec2i(tum3D::Vec2i& value) const;
	bool GetAsVec3i(tum3D::Vec3i& value) const;
	bool GetAsVec4i(tum3D::Vec4i& value) const;
	bool GetAsVec2f(tum3D::Vec2f& value) const;
	bool GetAsVec3f(tum3D::Vec3f& value) const;
	bool GetAsVec4f(tum3D::Vec4f& value) const;
	bool GetAsMat3f(tum3D::Mat3f& value) const;
	bool GetAsMat4f(tum3D::Mat4f& value) const;

	// output to ostream
	friend std::ostream& operator<<(std::ostream& o, const ConfigEntry& entry);

private:
	template<typename T> bool GetValue(const std::string& func, T& value, size_t index = 0) const;
	template<typename T> bool Get(const std::string& func, T& value) const;
	template<int N, typename T> bool GetVec(const std::string& func, Vec<N, T>& value) const;
	template<int M, int N, typename T> bool GetMat(const std::string& func, Mat<M, N, T>& value) const;

	std::string m_name;
	std::vector<std::string> m_values;
};

// A section of entries, started by: [section name]
class ConfigSection
{
public:
	ConfigSection(const std::string& name);

	const std::string& GetName() const { return m_name; }

	      std::vector<ConfigEntry>& GetEntries()       { return m_entries; }
	const std::vector<ConfigEntry>& GetEntries() const { return m_entries; }

	size_t GetEntryCount() const { return m_entries.size(); }

	void AddEntry(const ConfigEntry& entry) { m_entries.push_back(entry); }

	friend std::ostream& operator<<(std::ostream& o, const ConfigSection& section);

private:
	std::string m_name;
	std::vector<ConfigEntry> m_entries;
};

// A configuration file, containing any number of sections and entries.
// The section and entry order is preserved when writing to/reading from file.
// There may be multiple sections with the same name.
// All entries must belong to a section (i.e. no entries before the first section are allowed).
// A ';' or '#' marks the rest of the line as a comment.
// TODO provide a way to write comments?
class ConfigFile
{
public:
	bool Read(const std::string& filename);
	bool Write(const std::string& filename) const;

	      std::vector<ConfigSection>& GetSections()       { return m_sections; }
	const std::vector<ConfigSection>& GetSections() const { return m_sections; }

	size_t GetSectionCount() const { return m_sections.size(); }

	void AddSection(const ConfigSection& section) { m_sections.push_back(section); }

private:
	std::vector<ConfigSection> m_sections;
};


#endif
