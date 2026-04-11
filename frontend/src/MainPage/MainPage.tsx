import InputSection from "@/MainPage/InputSection/InputSection"
import SearchSection from "@/MainPage/SearchSection/SearchSection"

export default function MainPage(){
    return(
        <div className=" flex w-full h-full">
            <InputSection />
            <SearchSection />
        </div>
    )
}