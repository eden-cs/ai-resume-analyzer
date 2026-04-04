// Props this component receives from UploadPage
interface Props {
    message: string // error message to display
}

export default function ErrorMEssage({ message}: Props) {
    return (
        // Row with a dot and text
        <div className = "flex items-center gap-[5px] mt-2">

            {/* Error dot */}
            <div className = "w-[5px] h-[5px] rounded-full bg-error flex-shink-0" />

            {/* Error message */}
            <p className = "font-body font-medium text-[11px] text-error">
                {message}
            </p>
        </div>
    )
}