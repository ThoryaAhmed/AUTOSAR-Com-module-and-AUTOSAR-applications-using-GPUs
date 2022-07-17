/***************************************************************************************************
**

FileName:                                   Platform_Types.h                                   
AUTOSAR Version:                            4.2.2
																								 **
***************************************************************************************************/
#ifndef PLATFORM_TYPES_H
#define PLATFORM_TYPES_H
/***************************************************************************************************
**                                          Includes                                              **
****************************************************************************************************/
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>



/***************************************************************************************************
**                                        Types Declarations                                      **
****************************************************************************************************/
typedef  								signed char 							   sint8;
typedef 								unsigned char 							 	uint8;
typedef         						 short 										sint16;
typedef 								unsigned short 								uint16;
typedef      						    int                                         sint32;
typedef 								unsigned int								uint32;
typedef          						long long                                   sint64;
typedef 								unsigned long long 						    uint64;
typedef          						float                                       float32;
typedef 								double                  			        float64;
typedef                                  bool                                       boolean;
/****************************************************************************************************/

#endif
