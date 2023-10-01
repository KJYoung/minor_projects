import styled from "styled-components";

interface BtnProps {
    handler: () => void,
    disabled?: boolean
};

interface EditBtnProps extends BtnProps{

};

export const EditBtn = ({ handler, disabled } : EditBtnProps ) => {
    return <EditBtnWrapper onClick={handler}>
        <span>수정</span>
    </EditBtnWrapper>
};

export const EditCompleteBtn = ({ handler, disabled } : EditBtnProps ) => {
    return <EditCompleteBtnWrapper onClick={() => !disabled && handler()}>
        <span>수정 완료</span>
    </EditCompleteBtnWrapper>
};

interface DeleteBtnProps extends BtnProps {
    confirmText?: string
};

export const DeleteBtn = ({ handler, confirmText } : DeleteBtnProps ) => {
    return <DeleteBtnWrapper onClick={async () => {
        if(window.confirm(confirmText ? confirmText : '정말 삭제하시겠습니까?')){
            handler();
        }
    }}>
        <span>삭제</span>
    </DeleteBtnWrapper>
};

const AbstractBtn = styled.div`
    width: 50px;
    height: 25px;
    padding: 6px 3px;
    margin-left: 3px;
    
    cursor: pointer;
    border-radius: 12px;
    font-size: 12px;
    text-align: center;

    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
`;

const EditBtnWrapper = styled(AbstractBtn)`
    background-color: var(--ls-blue_gray);
`;
const EditCompleteBtnWrapper = styled(AbstractBtn)`
    width: 70px;
    background-color: var(--ls-blue_gray);
`;
const DeleteBtnWrapper = styled(AbstractBtn)`
    background-color: var(--ls-red_lighter);
`;