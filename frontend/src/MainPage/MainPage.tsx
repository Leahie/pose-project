import InputSection from "@/MainPage/InputSection/InputSection.tsx"
import SearchSection from "@/MainPage/SearchSection/SearchSection.tsx"

export default function MainPage(){
    return(
        <div className=" flex w-full h-full">
            <InputSection />
            <SearchSection />
        </div>
    )
}