import { TagElement, selectTag } from "../../store/slices/tag";
import { TagBubbleCompact } from "../general/TagBubble";
import { useSelector } from "react-redux";

interface TagDetailProps {
    selectedTag: TagElement | undefined
};


// containers/TagMain.tsx에서 사용되는 TagDetail 패널.
export const TagDetail = ({ selectedTag } : TagDetailProps) => {
    const { tagDetail } = useSelector(selectTag);

    return <>
        {selectedTag && <TagBubbleCompact color={selectedTag.color}>{selectedTag.name}</TagBubbleCompact>}
        {tagDetail && <div>
        {tagDetail.transaction.map((trxn) => {
            return <div key={trxn.id}>Trxn {trxn.memo}</div>
        })}  
        {tagDetail.todo.map((todo) => {
            return <div key={todo.id}>Todo {todo.name}</div>
        })}
        </div>}
    </>
}