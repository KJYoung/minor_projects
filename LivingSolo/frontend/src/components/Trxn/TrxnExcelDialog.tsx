import { BootstrapDialog, BootstrapDialogTitle } from "../general/Dialog";
import { styled } from "styled-components";
import { Button, DialogActions, DialogContent } from "@mui/material";
import { workbook, xlsxParser } from "../../utils/FileInterface";
import { useState } from "react";
import { DEFAULT_OPTION } from "../../utils/Constants";
import { IPropsActive, IPropsVarGrid } from "../../utils/Interfaces";
import { GetDateByAddingDatesFromExcel } from "../../utils/DateTime";

interface TrxnExcelDialogProps {
    open: boolean,
    handleClose: () => void,
    workbook: workbook,
};

export const TrxnExcelDialog = ({open, handleClose, workbook} : TrxnExcelDialogProps) => {
  const workbookSheetNames = workbook.SheetNames;
  const [sheetSelect, setSheetSelect] = useState<string>(DEFAULT_OPTION);
  const SheetContent = () => {
    const parsedSheet = xlsxParser(workbook.Sheets[sheetSelect]);
    const keys_ = Object.keys(parsedSheet[0] as any);
    let [keys, setKeys] = useState<string[]>([...keys_]);

    const colNameDivOnClickListener = (key: string, idx: number) => {
      if(keys.some((k) => k === key)){
        setKeys(keys.filter((k) => key !== k));
      }else{
        setKeys(k => {
          const new_list = [...k];
          new_list.splice(idx, 0, key);
          return new_list;
        });
      }
    }
    if(sheetSelect !== DEFAULT_OPTION){
        if(parsedSheet.length > 0){
            // console.log(parsedSheet);
            // console.log(keys);
            return <>
              <HeaderWrapper>
                {keys_.map((key, idx) => 
                  <HeaderNameDiv active={(keys.includes(key).toString())} key={key} onClick={() => colNameDivOnClickListener(key, idx)}>
                    {key}
                  </HeaderNameDiv>
                )}
              </HeaderWrapper>
              <RowWrapper gridlength={keys.length}>
                {keys.map((key) => <ColNameDiv key={key}>{key}</ColNameDiv>)}
              </RowWrapper>
              {parsedSheet.map((row: any) => {
                  return <RowWrapper gridlength={keys.length}>
                      {keys.map((key) => {
                          return <span key={key}>{row[key] && (key === '일자') ? GetDateByAddingDatesFromExcel(row[key]) : row[key]}</span>
                      })}
                  </RowWrapper>
              })}
            </> 
        }
    }
    return <></>
  }
  return <div>
    <BootstrapDialog
      onClose={handleClose}
      aria-labelledby="customized-dialog-title"
      open={open}
      fullWidth={true}
      maxWidth={false}
    >
      <DialogBody>
        <BootstrapDialogTitle id="customized-dialog-title" onClose={handleClose}>
          태그 설정
          <select value={sheetSelect} onChange={(e) => setSheetSelect(e.target.value)}>
              <option disabled value={DEFAULT_OPTION}>
                  - 시트를 선택하세요. -
              </option>
              {workbookSheetNames.map((sheetName) => <option value={sheetName} key={sheetName}>{sheetName}</option>)}
          </select>
        </BootstrapDialogTitle>
        <DialogContent dividers> 
          <span>
            * 주의사항: Header 바로 다음 행의 내용이 비어있는 경우 Column으로 인식되지 않습니다.
          </span>
          {sheetSelect !== DEFAULT_OPTION && <SheetContent />}  
        </DialogContent>
        <DialogActions>
          <Button autoFocus onClick={handleClose}>
            닫기
          </Button>
        </DialogActions>
      </DialogBody>
    </BootstrapDialog>
  </div>
};

const DialogBody = styled.div`
  width: 100%;
`;

const HeaderWrapper = styled.div`
  display: flex;
  text-align: center;
  padding: 2px;
`;

const RowWrapper = styled.div<IPropsVarGrid>`
  display: grid;
  grid-template-columns: ${props => ((props.gridlength) ? `repeat(${props.gridlength}, 1fr)` : '1fr')};
  text-align: center;

  padding: 2px;
`;

const HeaderNameDiv = styled.div<IPropsActive>`
  padding: 5px;
  margin-right: 12px;
  border-radius: 4px;
  background-color: ${props => ((props.active === 'true') ? 'var(--ls-green)' : 'var(--ls-gray)')};
  border: 1px solid black;
  color: ${props => ((props.active === 'true') ? 'var(--ls-black)' : 'var(--ls-white)')};
  cursor: pointer;
`;

const ColNameDiv = styled.div`
  border: 1px solid grey;
`;