// Props this component receives from ResultsPage
interface Props {
    keyword: string // the missing keyword to display in the pill
}

export default function KeywordPill({ keyword }: Props) {
    return (
        // Single span styled as a pill
        <span className = "font-body font-medium text-[11px] text-espresso bg-medium-tan border border-espresso/12 rounded-lg px-[11px] py[5px]">
            {keyword}
        </span>
    )
}