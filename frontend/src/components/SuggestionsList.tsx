// Props this component receives from the ResultsPage.
interface Props {
    suggestions: string[] // AI generated suggestions string from Gemini API.
}

export default function SuggestionsList({ suggestions }: Props) {
    const safeSuggestions = suggestions ?? []   // Handle the case where suggestions is null or undefined by defaulting to an empty array.

    return (
        <div className = "bg-taupe border border-espresso/12 rounded-xl p-5 mb-3">
           
            {/* Card label */}
            <p className = "font-body font-medium text-[10px] tracking-[0.13em] uppercase text-espresso opacity-60 mb-3">
                Suggestions
            </p>

            {/* List of suggestion bullet points */}
            <ul className = "flex flex-col gap-2.5">
                {safeSuggestions.map((suggestion, index) => (
                    <li key = {index} className = "flex items-start gap-2.5">

                        {/* Bullet point */}
                        <div className = "w-1.5 h-1.5 rounded-full bg-medium-brown flex-shrink-0 mt-[7px]" />
                        
                        {/* Suggestion text */}
                        <p className = "font-body text-[13px] text-espresso leading-relaxed">
                            {suggestion}
                        </p>
                    </li>
                ))}
            </ul>
        </div>
    )
}