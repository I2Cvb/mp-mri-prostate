include_directories(${PROJECT_SOURCE_DIR}/)

include_directories(${external_includes})

set(app_programs reg_dce reg_gt)

foreach(app ${app_programs})
    add_executable(${app} ${app}.cxx)
    target_link_libraries(${app}
                          ${external_libs})
endforeach()