find_package ( ITK REQUIRED )

if ( ITK_FOUND )
include( ${ITK_USE_FILE} )
endif( ITK_FOUND)

include_directories( ${ITK_LIBRARIES} )

set(external_libs ${ITK_LIBRARIES} "-lpthread")