from __future__ import annotations

import json
import math
import os
import logging
import pathlib
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Dict, Generator, IO, Any

from .tools import (
    generate_keys,
    chunk_size_and_padding,
    bytes_to_hex,
    is_usm,
    video_sink,
    audio_sink,
    slugify,
    pad_to_next_sector,
)
from .types import ChunkType, PayloadType, ElementType, OpMode
from .page import UsmPage, keyframes_from_seek_pages
from .chunk import UsmChunk
from .media import GenericVideo, GenericAudio, UsmVideo, UsmAudio


@dataclass
class UsmChannel:
    """Intermediate class for holding information on a UsmVideo
    and UsmAudio from a parsed Usm."""

    stream: List[Tuple[int, int]]
    header: UsmPage
    metadata: Optional[List[UsmPage]] = None


class Usm:
    def __init__(
        self,
        videos: List[UsmVideo],
        audios: Optional[List[UsmAudio]] = None,
        alphas: Optional[List[UsmVideo]] = None,
        key: Optional[int] = None,
        usm_crid: Optional[UsmPage] = None,
        version: Optional[int] = None,
    ) -> None:
        if len(videos) == 0:
            raise ValueError("No video given.")

        if audios is None:
            self.audios = []
        else:
            self.audios = audios
            self.audios.sort()

        if alphas is None:
            self.alphas = []
        else:
            self.alphas = alphas
            self.alphas.sort()

        self.version = version
        self.videos = videos
        self.videos.sort()

        self._usm_crid = usm_crid
        self._max_packet_size = 1

        logging.info(
            "Initialising USM",
            extra={
                "version": self.version,
                "is_key_given": key is not None,
                "is_usm_crid_given": usm_crid is not None,
                "num_videos": len(self.videos),
                "num_audios": len(self.audios),
                "num_alphas": len(self.alphas),
            },
        )

        self.max_frame = 0
        for vid in self.videos:
            self.max_frame = max(self.max_frame, len(vid))

        for aud in self.audios:
            self.max_frame = max(self.max_frame, len(aud))

        if key is None:
            self.video_key: Optional[bytes] = None
            self.audio_key: Optional[bytes] = None
        else:
            self.video_key, self.audio_key = generate_keys(key)

    @property
    def filename(self) -> str:
        if self._usm_crid is not None:
            crid_filename = self._usm_crid.get("filename")
            if crid_filename is not None:
                assert isinstance(crid_filename.val, str)
                return crid_filename.val.split("/")[-1]

        video_filename = self.videos[0].crid_page.get("filename")
        assert video_filename is not None and isinstance(video_filename.val, str)
        return video_filename.val.split("/")[-1].split(".")[0] + ".usm"

    def usm_crid_page(self, size_after_crid_part: Optional[int] = None) -> UsmPage:
        if self._usm_crid is not None:
            return self._usm_crid

        if size_after_crid_part is None:
            raise ValueError("Size after crid part not given.")

        crid = UsmPage("CRIUSF_DIR_STREAM")
        if self.version is not None:
            crid.update("fmtver", ElementType.I32, self.version)

        crid.update("filename", ElementType.STRING, self.filename)
        crid.update("filesize", ElementType.I32, 0x800 + size_after_crid_part)
        crid.update("datasize", ElementType.I32, 0)
        crid.update("stmid", ElementType.I32, 0)
        crid.update("chno", ElementType.I16, -1)
        crid.update("minchk", ElementType.I16, 1)

        # TODO: Find formula for minbuf
        minbuf = round(self._max_packet_size * 1.98746)
        minbuf += 0x10 - (minbuf % 0x10) if minbuf % 0x10 != 0 else 0
        crid.update("minbuf", ElementType.I32, minbuf)

        bitrate = 0
        for video in self.videos:
            bitrate += int(video.crid_page["avbps"].val)

        for audio in self.audios:
            bitrate += int(audio.crid_page["avbps"].val)

        crid.update("avbps", ElementType.I32, bitrate)

        return crid

    @classmethod
    def open(
        cls,
        filepath: Union[str, pathlib.Path],
        key: Optional[int] = None,
        encoding: str = "UTF-8",
    ) -> Usm:
        filesize = os.path.getsize(filepath)
        if filesize <= 0x20:
            raise ValueError(f"File {filepath} too small.")

        usmfile = open(filepath, "rb")
        filename = os.path.basename(filepath)
        logging.info(
            "Loading USM from file.",
            extra={
                "usm_name": filename,
                "size": filesize,
                "encoding": encoding,
                "is_key_given": key is not None,
            },
        )

        signature = usmfile.read(4)

        if not is_usm(signature):
            raise ValueError(f"Invalid file signature: {bytes_to_hex(signature)}")

        crids, video_channels, audio_channels, alpha_channels = _process_chunks(
            usmfile, filesize, encoding
        )

        # We don't need a mutex because of the GIL, but it feels dirty without one
        usmmutex = threading.Lock()
        videos = []
        audios = []
        alphas = []
        version: Optional[int] = None
        for channel_number, video_channel in video_channels.items():
            crid = [
                page
                for page in crids
                if (chno := page.get("chno")) is not None
                and chno.val == channel_number
                and (stmid := page.get("stmid")) is not None
                and stmid.val == 0x40534656  # @SFV
            ]

            if len(crid) == 0:
                raise ValueError(f"No crid page found for video ch {channel_number}.")
            if channel_number == 0:
                video_fmtver = crid[0].get("fmtver")
                if video_fmtver is not None and isinstance(video_fmtver.val, int):
                    version = video_fmtver.val

            # Create a factory that can replay the video stream
            def make_video_factory(usmfile, usmmutex, stream, keyframes):
                def video_factory():
                    return video_sink(usmfile, usmmutex, stream, keyframes)
                return video_factory

            videos.append(
                GenericVideo(
                    stream_factory=make_video_factory(
                        usmfile,
                        usmmutex,
                        video_channel.stream,
                        keyframes_from_seek_pages(video_channel.metadata),
                    ),
                    crid_page=crid[0],
                    header_page=video_channel.header,
                    metadata_pages=video_channel.metadata,
                    length=len(video_channel.stream),
                    channel_number=channel_number,
                )
            )

        for channel_number, audio_channel in audio_channels.items():
            crid = [
                page
                for page in crids
                if (chno := page.get("chno")) is not None
                and chno.val == channel_number
                and (stmid := page.get("stmid")) is not None
                and stmid.val == 0x40534641  # @SFA
            ]

            if len(crid) == 0:
                raise ValueError(f"No crid page found for audio ch {channel_number}.")

            # Create a factory that can replay the audio stream
            def make_audio_factory(usmfile, usmmutex, stream):
                def audio_factory():
                    return audio_sink(usmfile, usmmutex, stream)
                return audio_factory

            audios.append(
                GenericAudio(
                    stream_factory=make_audio_factory(
                        usmfile,
                        usmmutex,
                        audio_channel.stream,
                    ),
                    crid_page=crid[0],
                    header_page=audio_channel.header,
                    metadata_pages=audio_channel.metadata,
                    length=len(audio_channel.stream),
                    channel_number=channel_number,
                )
            )

        for channel_number, alpha_channel in alpha_channels.items():
            crid = [
                page
                for page in crids
                if (chno := page.get("chno")) is not None
                and chno.val == channel_number
                and (stmid := page.get("stmid")) is not None
                and stmid.val == 0x40414C50
            ]

            if len(crid) == 0:
                raise ValueError(f"No crid page found for alpha ch {channel_number}")

            # Create a factory that can replay the alpha video stream
            def make_alpha_factory(usmfile, usmmutex, stream, keyframes):
                def alpha_factory():
                    return video_sink(usmfile, usmmutex, stream, keyframes)
                return alpha_factory

            alphas.append(
                GenericVideo(
                    stream_factory=make_alpha_factory(
                        usmfile,
                        usmmutex,
                        alpha_channel.stream,
                        keyframes_from_seek_pages(alpha_channel.metadata),
                    ),
                    crid_page=crid[0],
                    header_page=alpha_channel.header,
                    metadata_pages=alpha_channel.metadata,
                    length=len(alpha_channel.stream),
                    channel_number=channel_number,
                    is_alpha=True,
                )
            )

        usm_crid = [
            page
            for page in crids
            if (chno := page.get("chno")) is not None and chno.val == -1
        ]
        if len(usm_crid) == 0:
            raise ValueError("No usm crid page found.")

        return cls(
            version=version,
            videos=videos,
            audios=audios,
            alphas=alphas,
            key=key,
            usm_crid=usm_crid[0],
        )

    @staticmethod
    def _page_to_dict(page: UsmPage) -> Dict[str, Any]:
        """Convert a UsmPage to a JSON-serializable dictionary."""
        entries = {}
        for key, element in page.dict.items():
            value = element.val
            if isinstance(value, (bytes, bytearray)):
                value = value.hex()
            elif isinstance(value, tuple) and len(value) == 1:
                value = value[0]
            entries[key] = {"type": element.type.name, "value": value}
        return {"name": page.name, "entries": entries}

    def demux(
        self,
        path: str,
        save_video: bool = True,
        save_audio: bool = True,
        save_alpha: bool = True,
        save_pages: bool = False,
        folder_name: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """Saves all videos, audios, alpha videos, pages (depending on configuration) of a Usm."""
        if folder_name is None:
            folder_name = self.filename

        folder_name = slugify(folder_name, allow_unicode=True)
        output = os.path.join(path, folder_name)
        if os.path.exists(output) and os.path.isfile(output):
            raise FileExistsError

        os.makedirs(output, exist_ok=True)

        videos = []
        audios = []
        alphas = []

        def save(usm_array, output_array, name, key):
            if len(usm_array) == 0:
                return

            logging.info(f"Saving {name}")
            mode = OpMode.NONE if key is None else OpMode.DECRYPT
            sub_output = os.path.join(output, name)
            if not os.path.exists(sub_output):
                os.mkdir(sub_output)

            for item in usm_array:
                filename = os.path.join(sub_output, item.filename)
                with open(filename, "wb") as f:
                    for packet in item.stream(mode, key):
                        f.write(packet if type(packet) is not tuple else packet[0])

                output_array.append(filename)

        if save_video:
            save(self.videos, videos, "videos", self.video_key)

        if save_audio:
            save(self.audios, audios, "audios", self.audio_key)

        if save_alpha:
            save(self.alphas, alphas, "alphas", self.video_key)

        if save_pages:
            logging.info("Saving pages")
            pages_output = os.path.join(output, "pages")
            os.makedirs(pages_output, exist_ok=True)

            # Save USM-level CRID page
            usm_pages_dir = os.path.join(pages_output, "usm")
            os.makedirs(usm_pages_dir, exist_ok=True)
            usm_crid_dict = self._page_to_dict(self.usm_crid)
            with open(os.path.join(usm_pages_dir, "crid.json"), "w", encoding="utf-8") as f:
                json.dump(usm_crid_dict, f, indent=2, ensure_ascii=False)

            # Save video pages
            if len(self.videos) > 0:
                videos_pages_dir = os.path.join(pages_output, "videos")
                os.makedirs(videos_pages_dir, exist_ok=True)
                for i, video in enumerate(self.videos):
                    channel_dir = os.path.join(videos_pages_dir, f"channel_{video.channel_number:03d}")
                    os.makedirs(channel_dir, exist_ok=True)

                    # Save CRID page
                    crid_dict = self._page_to_dict(video.crid_page)
                    with open(os.path.join(channel_dir, "crid.json"), "w", encoding="utf-8") as f:
                        json.dump(crid_dict, f, indent=2, ensure_ascii=False)

                    # Save header page
                    header_dict = self._page_to_dict(video.header_page)
                    with open(os.path.join(channel_dir, "header.json"), "w", encoding="utf-8") as f:
                        json.dump(header_dict, f, indent=2, ensure_ascii=False)

                    # Save metadata pages if present
                    if video.metadata_pages is not None and len(video.metadata_pages) > 0:
                        for j, metadata_page in enumerate(video.metadata_pages):
                            metadata_dict = self._page_to_dict(metadata_page)
                            metadata_filename = os.path.join(channel_dir, f"metadata_{j:03d}.json")
                            with open(metadata_filename, "w", encoding="utf-8") as f:
                                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

            # Save audio pages
            if len(self.audios) > 0:
                audios_pages_dir = os.path.join(pages_output, "audios")
                os.makedirs(audios_pages_dir, exist_ok=True)
                for i, audio in enumerate(self.audios):
                    channel_dir = os.path.join(audios_pages_dir, f"channel_{audio.channel_number:03d}")
                    os.makedirs(channel_dir, exist_ok=True)

                    # Save CRID page
                    crid_dict = self._page_to_dict(audio.crid_page)
                    with open(os.path.join(channel_dir, "crid.json"), "w", encoding="utf-8") as f:
                        json.dump(crid_dict, f, indent=2, ensure_ascii=False)

                    # Save header page
                    header_dict = self._page_to_dict(audio.header_page)
                    with open(os.path.join(channel_dir, "header.json"), "w", encoding="utf-8") as f:
                        json.dump(header_dict, f, indent=2, ensure_ascii=False)

                    # Save metadata pages if present
                    if audio.metadata_pages is not None and len(audio.metadata_pages) > 0:
                        for j, metadata_page in enumerate(audio.metadata_pages):
                            metadata_dict = self._page_to_dict(metadata_page)
                            metadata_filename = os.path.join(channel_dir, f"metadata_{j:03d}.json")
                            with open(metadata_filename, "w", encoding="utf-8") as f:
                                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

            # Save alpha pages
            if len(self.alphas) > 0:
                alphas_pages_dir = os.path.join(pages_output, "alphas")
                os.makedirs(alphas_pages_dir, exist_ok=True)
                for i, alpha in enumerate(self.alphas):
                    channel_dir = os.path.join(alphas_pages_dir, f"channel_{alpha.channel_number:03d}")
                    os.makedirs(channel_dir, exist_ok=True)

                    # Save CRID page
                    crid_dict = self._page_to_dict(alpha.crid_page)
                    with open(os.path.join(channel_dir, "crid.json"), "w", encoding="utf-8") as f:
                        json.dump(crid_dict, f, indent=2, ensure_ascii=False)

                    # Save header page
                    header_dict = self._page_to_dict(alpha.header_page)
                    with open(os.path.join(channel_dir, "header.json"), "w", encoding="utf-8") as f:
                        json.dump(header_dict, f, indent=2, ensure_ascii=False)

                    # Save metadata pages if present
                    if alpha.metadata_pages is not None and len(alpha.metadata_pages) > 0:
                        for j, metadata_page in enumerate(alpha.metadata_pages):
                            metadata_dict = self._page_to_dict(metadata_page)
                            metadata_filename = os.path.join(channel_dir, f"metadata_{j:03d}.json")
                            with open(metadata_filename, "w", encoding="utf-8") as f:
                                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

        return videos, audios

    def _generate_prestream_chunks(
        self,
        stream_filesize: int,
        keyframe_index_and_offsets: dict,
        encoding: str,
    ) -> Generator[UsmChunk, None, None]:
        header_metadata_chunks = []
        header_metadata_size = 0
        for chunk, position in _generate_header_metadata_chunks(
            self.videos, self.audios, keyframe_index_and_offsets, encoding
        ):
            header_metadata_chunks.append(chunk)
            header_metadata_size = position

        usm_crid_page = self.usm_crid_page(
            0x800 + header_metadata_size + stream_filesize
        )
        pages = [usm_crid_page]
        for video in self.videos:
            pages.append(video.crid_page)

        for audio in self.audios:
            pages.append(audio.crid_page)

        yield UsmChunk(
            chunk_type=ChunkType.INFO,
            payload_type=PayloadType.HEADER,
            payload=pages,
            padding=pad_to_next_sector(position=0),
            encoding=encoding,
        )

        for chunk in header_metadata_chunks:
            yield chunk

    def chunks(
        self, mode: OpMode = OpMode.NONE, encoding: str = "UTF-8"
    ) -> Generator[UsmChunk, None, None]:
        # Collect statistics in first pass without buffering
        stats = collect_mux_stats(
            self.max_frame,
            self.videos,
            self.audios,
            mode,
            self.video_key,
            self.audio_key,
        )
        self._max_packet_size = stats.max_chunk

        # Generate prestream chunks (header, metadata, etc.)
        for chunk in self._generate_prestream_chunks(
            stream_filesize=stats.total_size,
            keyframe_index_and_offsets=stats.keyframes,
            encoding=encoding,
        ):
            yield chunk

        # Stream chunks directly without buffering to disk
        for chunk in iter_mux_stream(
            self.max_frame,
            self.videos,
            self.audios,
            mode,
            self.video_key,
            self.audio_key,
        ):
            yield chunk

    def stream(
        self, mode: OpMode = OpMode.NONE, encoding: str = "UTF-8"
    ) -> Generator[bytes, None, None]:
        # Collect statistics in first pass without buffering
        stats = collect_mux_stats(
            self.max_frame,
            self.videos,
            self.audios,
            mode,
            self.video_key,
            self.audio_key,
        )
        self._max_packet_size = stats.max_chunk

        # Generate prestream chunks (header, metadata, etc.) and yield their bytes
        for chunk in self._generate_prestream_chunks(
            stream_filesize=stats.total_size,
            keyframe_index_and_offsets=stats.keyframes,
            encoding=encoding,
        ):
            yield chunk.pack()

        # Stream chunks directly and yield their packed bytes
        for chunk in iter_mux_stream(
            self.max_frame,
            self.videos,
            self.audios,
            mode,
            self.video_key,
            self.audio_key,
        ):
            yield chunk.pack()


def _chunk_helper(default_dict_ch: Dict[int, UsmChannel], chunk: UsmChunk, offset: int):
    """Helper function for _process_chunks. Fills default_dict_ch with information about
    the passed chunk and offset."""
    if chunk.payload_type == PayloadType.STREAM:
        default_dict_ch[chunk.channel_number].stream.append(
            (offset + chunk.payload_offset, len(chunk.payload))
        )
    elif chunk.payload_type == PayloadType.SECTION_END:
        logging.debug(
            f"{chunk.chunk_type} section end",
            extra={
                "payload": bytes_to_hex(chunk.payload)
                if isinstance(chunk.payload, bytes)
                else chunk.payload,
                "offset": offset,
            },
        )
    elif chunk.payload_type == PayloadType.HEADER:
        default_dict_ch[chunk.channel_number].header = chunk.payload[0]
    elif chunk.payload_type == PayloadType.METADATA:
        default_dict_ch[chunk.channel_number].metadata = chunk.payload
    else:
        raise ValueError(f"Unknown payload type: {chunk.payload_type}")


def _process_chunks(
    usmfile: IO,
    filesize: int,
    encoding: str,
) -> Tuple[
    List[UsmPage], Dict[int, UsmChannel], Dict[int, UsmChannel], Dict[int, UsmChannel]
]:
    """Helper function that reads all the chunks in a USM file and returns a tuple of
    1. A list of USM pages about the contents of the USM file.
    2. A dictionary of USM video channels.
    3. A dictionary of USM audio channels.
    4. A dictionary of USM alpha video channels."""
    crids: List[UsmPage] = []
    video_ch: Dict[int, UsmChannel] = defaultdict(
        lambda: UsmChannel(stream=[], header=UsmPage(""))
    )
    audio_ch: Dict[int, UsmChannel] = defaultdict(
        lambda: UsmChannel(stream=[], header=UsmPage(""))
    )
    alpha_ch: Dict[int, UsmChannel] = defaultdict(
        lambda: UsmChannel(stream=[], header=UsmPage(""))
    )

    usmfile.seek(0, 0)
    while filesize > usmfile.tell():
        # Peek to read chunk's true size and _padding.
        temp_buf = usmfile.read(0x20)
        usmfile.seek(-0x20, 1)

        chunk_size, chunk_padding = chunk_size_and_padding(temp_buf)
        offset = usmfile.tell()

        # Read chunk payload and the 0x20 byte chunk header. Then skip _padding.
        data = usmfile.read(chunk_size + 0x20)
        usmfile.seek(chunk_padding, 1)

        try:
            chunk = UsmChunk.from_bytes(data, encoding=encoding)
        except ValueError as e:
            # If in debug mode, continue gathering information about the problematic usm
            if logging.root.level <= logging.DEBUG:
                logging.error(e)
                continue
            else:
                raise

        if chunk.chunk_type is ChunkType.INFO:
            if isinstance(chunk.payload, list):
                crids.extend(chunk.payload)
            else:
                logging.warning(
                    "Received info chunk payload that's not a list",
                    extra={"payload": chunk.payload},
                )
        # Video chunk
        elif chunk.chunk_type is ChunkType.VIDEO:
            _chunk_helper(video_ch, chunk, offset)

        # Alpha chunk
        elif chunk.chunk_type is ChunkType.ALPHA:
            _chunk_helper(alpha_ch, chunk, offset)

        # Audio chunk
        elif chunk.chunk_type is ChunkType.AUDIO:
            _chunk_helper(audio_ch, chunk, offset)

    return crids, video_ch, audio_ch, alpha_ch


def _generate_header_metadata_chunks(
    videos: List[UsmVideo],
    audios: List[UsmAudio],
    keyframe_index_and_offsets: Dict[int, List[Tuple[int, int]]],
    encoding: str,
) -> Generator[Tuple[UsmChunk, int], None, None]:
    current_position = 0
    # ========= YIELD HEADER PAGE CHUNKS ==========

    for video in videos:
        chunk = UsmChunk(
            chunk_type=ChunkType.VIDEO,
            payload_type=PayloadType.HEADER,
            payload=[video.header_page],
            padding=0x18,  # Based from real USMs
            channel_number=video.channel_number,
            encoding=encoding,
        )
        current_position += len(chunk)
        yield chunk, current_position

    for audio in audios:
        chunk = UsmChunk(
            chunk_type=ChunkType.AUDIO,
            payload_type=PayloadType.HEADER,
            payload=[audio.header_page],
            padding=0x8,  # Based from real USMs
            channel_number=audio.channel_number,
            encoding=encoding,
        )
        current_position += len(chunk)
        yield chunk, current_position

    # ========== YIELD HEADER END CHUNKS ==========

    header_end_payload = "#HEADER END     ===============".encode("UTF-8") + bytes(1)
    for video in videos:
        chunk = UsmChunk(
            chunk_type=ChunkType.VIDEO,
            payload_type=PayloadType.SECTION_END,
            payload=header_end_payload,
            padding=0,  # Based from real USMs
            channel_number=video.channel_number,
            encoding=encoding,
        )
        current_position += len(chunk)
        yield chunk, current_position

    for audio in audios:
        chunk = UsmChunk(
            chunk_type=ChunkType.AUDIO,
            payload_type=PayloadType.SECTION_END,
            payload=header_end_payload,
            padding=0,  # Based from real USMs
            channel_number=audio.channel_number,
            encoding=encoding,
        )
        current_position += len(chunk)
        yield chunk, current_position

    # ========== PROCESS METADATA CHUNKS ==========

    def metadata_pad(size: int) -> int:
        # TODO: Find cases where this does not hold for metadata chunks
        if size <= 0xF0:
            return 0xF0 - size
        else:
            return math.ceil(size / 0x8) * 0x8 - size

    metadata_section_size = 0
    metadata_section_chunks_vid = []
    metadata_section_chunks_aud = []
    metadata_section_chunks_sec_end = []

    for video in videos:
        if video.metadata_pages is None:
            index_and_offsets = keyframe_index_and_offsets[video.channel_number]
            metadata_pages: List[UsmPage] = []
            for index, offset in index_and_offsets:
                page = UsmPage("VIDEO_SEEKINFO")
                # ofs_byte is modified later
                page.update("ofs_byte", ElementType.I64, offset)
                page.update("ofs_frmid", ElementType.U32, index)
                page.update("num_skip", ElementType.U16, 0)
                page.update("resv", ElementType.U16, 0)
                metadata_pages.append(page)
        else:
            metadata_pages = video.metadata_pages

        chunk = UsmChunk(
            chunk_type=ChunkType.VIDEO,
            payload_type=PayloadType.METADATA,
            payload=metadata_pages,
            padding=metadata_pad,
            channel_number=video.channel_number,
            encoding=encoding,
        )
        metadata_section_size += len(chunk)
        metadata_section_chunks_vid.append(chunk)

    for audio in audios:
        if audio.metadata_pages is None:
            continue

        chunk = UsmChunk(
            chunk_type=ChunkType.AUDIO,
            payload_type=PayloadType.METADATA,
            payload=audio.metadata_pages,
            padding=metadata_pad,
            channel_number=audio.channel_number,
            encoding=encoding,
        )
        metadata_section_size += len(chunk)
        metadata_section_chunks_aud.append(chunk)

    metadata_end_payload = bytes("#METADATA END   ===============", "UTF-8") + bytes(1)
    for video in videos:
        chunk = UsmChunk(
            chunk_type=ChunkType.VIDEO,
            payload_type=PayloadType.SECTION_END,
            payload=metadata_end_payload,
            padding=0,  # Based from real USMs
            channel_number=video.channel_number,
            encoding=encoding,
        )
        metadata_section_size += len(chunk)
        metadata_section_chunks_sec_end.append(chunk)

    for audio in audios:
        if audio.metadata_pages is None:
            continue

        chunk = UsmChunk(
            chunk_type=ChunkType.AUDIO,
            payload_type=PayloadType.SECTION_END,
            payload=metadata_end_payload,
            padding=0,  # Based from real USMs
            channel_number=audio.channel_number,
            encoding=encoding,
        )
        metadata_section_size += len(chunk)
        metadata_section_chunks_sec_end.append(chunk)

    # ========= YIELD METADATA CHUNKS ==========

    for chunk in metadata_section_chunks_vid:
        payload = chunk.payload
        if isinstance(payload, bytes):
            raise ValueError("Video metadata is not list of pages.")
        else:
            # Add 0x800(crid chunks and _padding) and the size of the entire
            # metadata section to offsets of stream file
            for metadata in payload:
                offset = metadata["ofs_byte"].val
                offset += 0x800 + current_position + metadata_section_size
                metadata.update("ofs_byte", ElementType.I64, offset)

        yield chunk, current_position + metadata_section_size

    for chunk in metadata_section_chunks_aud:
        yield chunk, current_position + metadata_section_size

    for chunk in metadata_section_chunks_sec_end:
        yield chunk, current_position + metadata_section_size


@dataclass
class MuxStats:
    """Statistics collected from a first pass over the muxed stream."""
    total_size: int
    max_chunk: int
    keyframes: Dict[int, List[Tuple[int, int]]]


def collect_mux_stats(
    max_frames: int,
    videos: List[UsmVideo],
    audios: List[UsmAudio],
    mode: OpMode = OpMode.NONE,
    video_key: Optional[bytes] = None,
    audio_key: Optional[bytes] = None,
) -> MuxStats:
    """Collect statistics from the muxed stream without writing to disk.

    This function performs a first pass over all video and audio chunks to
    compute total size, maximum chunk size, and keyframe offsets. It mirrors
    the interleaving logic of iter_mux_stream but never writes packets to disk.

    Returns:
        MuxStats containing total_size, max_chunk, and keyframes dictionary.
    """
    videos_iter: List[Generator[Tuple[List[UsmChunk], bool], None, None]] = [
        vid.chunks(mode=mode, key=video_key) for vid in videos
    ]
    audios_iter: List[Generator[List[UsmChunk], None, None]] = [
        aud.chunks(mode=mode, key=audio_key) for aud in audios
    ]

    keyframe_index_and_offsets: Dict[int, List[Tuple[int, int]]] = defaultdict(
        lambda: list()
    )
    max_packet_size = 1
    current_offset = 0

    for index in range(max_frames):
        finished_video_iters = []
        finished_audio_iters = []

        # Process videos generators
        for i, vid_gen in enumerate(videos_iter):
            try:
                chunks, is_keyframe = next(vid_gen)
                if is_keyframe:
                    keyframe_index_and_offsets[chunks[0].channel_number].append(
                        (index, current_offset)
                    )

                for chunk in chunks:
                    chunk_size = len(chunk)
                    max_packet_size = max(chunk_size, max_packet_size)
                    current_offset += chunk_size
            except StopIteration:
                finished_video_iters.append(i)
                continue

        # Process audios generators
        for i, aud_gen in enumerate(audios_iter):
            try:
                chunks = next(aud_gen)
                for chunk in chunks:
                    chunk_size = len(chunk)
                    max_packet_size = max(chunk_size, max_packet_size)
                    current_offset += chunk_size
            except StopIteration:
                finished_audio_iters.append(i)
                continue

        # Remove finished generators
        videos_iter = [
            vid for i, vid in enumerate(videos_iter) if i not in finished_video_iters
        ]
        audios_iter = [
            aud for i, aud in enumerate(audios_iter) if i not in finished_audio_iters
        ]

    return MuxStats(
        total_size=current_offset,
        max_chunk=max_packet_size,
        keyframes=keyframe_index_and_offsets,
    )


def iter_mux_stream(
    max_frames: int,
    videos: List[UsmVideo],
    audios: List[UsmAudio],
    mode: OpMode = OpMode.NONE,
    video_key: Optional[bytes] = None,
    audio_key: Optional[bytes] = None,
) -> Generator[UsmChunk, None, None]:
    """Stream chunks directly without buffering to disk.

    This function replays the same scheduling logic as collect_mux_stats
    but yields each UsmChunk as soon as it is produced. Because media streams
    are now replayable via stream factories, this second pass can regenerate
    payloads without buffering.

    Yields:
        UsmChunk objects in the order they should appear in the muxed stream.
    """
    videos_iter: List[Generator[Tuple[List[UsmChunk], bool], None, None]] = [
        vid.chunks(mode=mode, key=video_key) for vid in videos
    ]
    audios_iter: List[Generator[List[UsmChunk], None, None]] = [
        aud.chunks(mode=mode, key=audio_key) for aud in audios
    ]

    for index in range(max_frames):
        finished_video_iters = []
        finished_audio_iters = []

        # Process videos generators
        for i, vid_gen in enumerate(videos_iter):
            try:
                chunks, is_keyframe = next(vid_gen)
                # Yield all chunks for this video frame
                for chunk in chunks:
                    yield chunk
            except StopIteration:
                finished_video_iters.append(i)
                continue

        # Process audios generators
        for i, aud_gen in enumerate(audios_iter):
            try:
                chunks = next(aud_gen)
                # Yield all chunks for this audio frame
                for chunk in chunks:
                    yield chunk
            except StopIteration:
                finished_audio_iters.append(i)
                continue

        # Remove finished generators
        videos_iter = [
            vid for i, vid in enumerate(videos_iter) if i not in finished_video_iters
        ]
        audios_iter = [
            aud for i, aud in enumerate(audios_iter) if i not in finished_audio_iters
        ]
