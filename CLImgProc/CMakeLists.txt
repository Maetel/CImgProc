set_as_lib()
link_CImgProc()
link_opencl()

set(INSTALL_DEBUG_DEST "${CLIMGPROC_LIB}/Debug")
set(INSTALL_RELEASE_DEST "${CLIMGPROC_LIB}/Release")
set(INSTALL_RELWITHDEBINFO_DEST "${CLIMGPROC_LIB}/RelWithDebInfo")

target_include_directories(${current_project} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(DIRECTORY DESTINATION ${CLIMGPROC_LIB})
install(TARGETS ${current_project}
    CONFIGURATIONS Debug
    DESTINATION ${INSTALL_DEBUG_DEST}
    )
install(TARGETS ${current_project}
    CONFIGURATIONS Release
    DESTINATION ${INSTALL_RELEASE_DEST}
    )
install(TARGETS ${current_project}
    CONFIGURATIONS RelWithDebInfo
    DESTINATION ${INSTALL_RELWITHDEBINFO_DEST}
    )

#copy pdb files if exists for shared libraries
if(CLImgProc_build_shared)
    install(FILES $<TARGET_PDB_FILE:${current_project}>
        CONFIGURATIONS Debug
        DESTINATION ${INSTALL_DEBUG_DEST}
        OPTIONAL
        )
    install(FILES $<TARGET_PDB_FILE:${current_project}>
        CONFIGURATIONS Release
        DESTINATION ${INSTALL_RELEASE_DEST}
        OPTIONAL
        )
    install(FILES $<TARGET_PDB_FILE:${current_project}>
        CONFIGURATIONS RelWithDebInfo
        DESTINATION ${INSTALL_RELWITHDEBINFO_DEST}
        OPTIONAL
        )
endif()

FILE(GLOB headers "*.h" "*.hpp")
FILE(GLOB cl_srcs "cl_srcs/*.cl")
install(FILES ${headers} DESTINATION "${CLIMGPROC_INCLUDE}")
install(FILES ${cl_srcs} DESTINATION "${CLIMGPROC_INCLUDE}/cl_srcs")
#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config_impl.hpp DESTINATION "${include_dest}")