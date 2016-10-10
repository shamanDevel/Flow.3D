xcopy /y C:\Code\cudaCompress\bin\x64\Release\cudaCompress.dll dll\Release
xcopy /y C:\Code\cudaCompress\bin\x64\Release\cudaCompress.lib lib
xcopy /y C:\Code\cudaCompress\bin\x64\Debug\cudaCompressd.dll dll\Debug
xcopy /y C:\Code\cudaCompress\bin\x64\Debug\cudaCompressd.lib lib
xcopy /y C:\Code\cudaCompress\bin\x64\Debug\*.pdb lib
rd /s /q include\cudaCompress
xcopy /i /s C:\Code\cudaCompress\include\cudaCompress include\cudaCompress