import React from 'react'

import Image from 'next/image';


export default function HomeStarter() {
  return (
    <div className='min-h-screen flex flex-col items-center pt-8'> 
      <h1 className='text-3xl font-bold mb-4'>Information Retrieval Project</h1> 
      <p className='mt-2 font-semibold text-lg'>Ask questions about BITS Pilani information</p> 

      {/* Commenting out example query boxes as requested */}
      {/* 
      <div className='flex flex-wrap justify-center gap-4 mt-10'> 
        <div className='border-[1.2px] border-orange-700 text-gray-600 h-auto min-h-20 w-44 flex-col gap-3 rounded-xl shadow-sm hover:cursor-pointer hover:bg-orange-100 hover:text-orange-600 transition-colors p-4 flex items-center justify-center'>
          <span className='text-sm font-medium text-center '>How are PhD stipends handled?</span>
        </div>
        <div className='border-[1.2px] border-orange-700 text-gray-600 h-auto min-h-20 w-44 flex-col gap-3 rounded-xl shadow-sm hover:cursor-pointer hover:bg-orange-100 hover:text-orange-600 transition-colors p-4 flex items-center justify-center'>
          <span className='text-sm font-medium text-center '>What are the library rules?</span>
        </div>
        <div className='border-[1.2px] border-orange-700 text-gray-600 h-auto min-h-20 w-44 flex-col gap-3 rounded-xl shadow-sm hover:cursor-pointer hover:bg-orange-100 hover:text-orange-600 transition-colors p-4 flex items-center justify-center'>
          <span className='text-sm font-medium text-center '>Tell me about hostel allocation.</span>
        </div>
        <div className='border-[1.2px] border-orange-700 text-gray-600 h-auto min-h-20 w-44 flex-col gap-3 rounded-xl shadow-sm hover:cursor-pointer hover:bg-orange-100 hover:text-orange-600 transition-colors p-4 flex items-center justify-center'>
          <span className='text-sm font-medium text-center'>What is the process for course registration?</span>
        </div>
      </div>
      */}
    </div>
  )
}